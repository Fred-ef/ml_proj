"""Run independent tasks across worker processes

Configurations share no state, so they can be evaluated in parallel. Processes
instead of threads because of the GIL; each worker uses one BLAS thread to
avoid oversubscription
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Callable, Sequence

# thread-limit env vars of the common BLAS/OpenMP backends
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
    """Apply func(*args) to every tuple in arg_tuples across processes

    Results come back in completion order. on_result(done, result), if given,
    is called after each task to report progress. With n_core == 1 everything
    runs in the main process, without a pool. On the first error the tasks not
    yet started are cancelled and the error is raised once the running ones end

    func and arg_tuples must be picklable (module-level functions, plain data,
    numpy arrays)
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
        try:
            for fut in as_completed(futures):     # finish order
                result = fut.result()
                done += 1
                results.append(result)
                if on_result:
                    on_result(done, result)
        except BaseException:
            # Leaving the with-block waits for every submitted task: on an error
            # (or Ctrl+C) cancel the queued ones and wait only for the running ones
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    return results
