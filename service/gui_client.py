"""In-process API client used by the NiceGUI console (service/gui.py).

The GUI is a CLIENT of the very same FastAPI app it is mounted on. Rather than
importing service internals and re-implementing the route logic, it speaks the
real HTTP contract through an httpx AsyncClient wired to the app via
ASGITransport — no socket, no port, no CORS: the request is dispatched straight
into `app`, exactly the way tests/test_service.py's TestClient already does it.

Net effect: every submit/read goes through the SAME validation and route logic
in service/app.py, so the GUI and the HTTP API can never drift apart, and there
is zero duplicated business logic here — only transport + error shaping.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from service.app import app

# base_url is arbitrary (ASGITransport ignores the network host); it only needs
# to be a valid absolute URL so httpx can build requests.
_transport = httpx.ASGITransport(app=app)
_BASE = "http://gui.local"


class ApiError(Exception):
    """A non-2xx response from the service, carrying status + human detail so the
    UI can show exactly what the API rejected (e.g. a 422 validation message)."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def _detail(resp: httpx.Response) -> str:
    """Best-effort extraction of FastAPI's error `detail` for display."""
    try:
        body = resp.json()
    except Exception:
        return resp.text or resp.reason_phrase
    if isinstance(body, dict) and "detail" in body:
        d = body["detail"]
        return d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)
    return json.dumps(body, ensure_ascii=False)


async def _request(method: str, url: str, **kw: Any) -> httpx.Response:
    async with httpx.AsyncClient(transport=_transport, base_url=_BASE) as client:
        resp = await client.request(method, url, **kw)
    if resp.status_code >= 400:
        raise ApiError(resp.status_code, _detail(resp))
    return resp


# --------------------------------------------------------------------- jobs
async def post_job(body: dict) -> dict:
    """POST /jobs — returns the created JobRecord.public() dict (202)."""
    resp = await _request("POST", "/jobs", json=body)
    return resp.json()


async def list_jobs() -> list[dict]:
    resp = await _request("GET", "/jobs")
    return resp.json()


async def get_job(job_id: str) -> dict:
    resp = await _request("GET", f"/jobs/{job_id}")
    return resp.json()


# --------------------------------------------------------------------- runs
async def get_run(task: str, run_id: str) -> dict:
    """GET /runs/{task}/{run_id} — {'config': ..., 'summary': ...}."""
    resp = await _request("GET", f"/runs/{task}/{run_id}")
    return resp.json()


async def get_index(task: str) -> list[dict]:
    """GET /runs?task=... — the flat comparison table (one row per run)."""
    resp = await _request("GET", "/runs", params={"task": task})
    return resp.json()


async def get_plot_bytes(task: str, run_id: str) -> Optional[bytes]:
    """PNG bytes of the learning curve, or None when the run has none
    (select mode / 404) so the UI can show a graceful fallback."""
    try:
        resp = await _request("GET", f"/runs/{task}/{run_id}/plot")
    except ApiError as e:
        if e.status == 404:
            return None
        raise
    return resp.content
