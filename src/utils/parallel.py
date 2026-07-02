"""Run independent tasks across worker processes

Since each config's k-fold CV/multi-seed trial is independent (no shared
state), we can parallelize them.

Design decisions:
  - processes instead of threads: CPython's GIL lets only one thread run
    Python bytecode at a time, so threads would serialize, while separate
    processes have separate interpreters.
  - one BLAS thread per worker and we spawn workers on start-up to avoid
    BLAS oversubscription.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Callable, Sequence

# Thread-limit env vars honored by the common BLAS/OpenMP backends.
_BLAS_ENV_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def resolve_workers(n_core: int | None, n_tasks: int) -> int:
    """Map a config's ``n_core`` to a concrete worker-process count:

    - None or 1 => 1 (run in the main process)
    - -1 => all cores
    - -2 => all but one core
    """
    if n_core in (None, 1):
        return 1
    cpu = os.cpu_count() or 1
    workers = max(1, cpu + 1 + n_core) if n_core < 0 else n_core
    return max(1, min(workers, n_tasks))


def parallel_map(func: Callable, arg_tuples: Sequence[tuple], n_core: int | None,
                 on_result: Callable[[int, object], None] | None = None) -> list:
    """Apply func(*args) for every args in arg_tuples across processes

    Returns the results in COMPLETION order.
    if given, on_result(done, result) is invoked after each task for
    progress reporting

    NOTE: With n_core == 1 the work runs in-process with no pool

    func and everything in arg_tuples must be picklable
    (module-level functions, plain data, numpy arrays)
    """
    workers = resolve_workers(n_core, len(arg_tuples))

    if workers == 1:
        results = []
        for i, args in enumerate(arg_tuples, 1):
            result = func(*args)
            results.append(result)
            if on_result:
                on_result(i, result)
        return results

    # Keep each spawned worker from oversubscribing BLAS threads
    for var in _BLAS_ENV_VARS:
        os.environ.setdefault(var, "1")
    ctx = get_context("spawn")

    results, done = [], 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(func, *args) for args in arg_tuples]
        for fut in as_completed(futures):     # finish order
            result = fut.result()
            done += 1
            results.append(result)
            if on_result:
                on_result(done, result)
    return results
