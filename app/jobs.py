"""Background job runner for long simulations.

Jobs run in background threads. Progress is tracked in SQLite.
Clients poll GET /jobs/{id} for status.
"""
import threading
import time
import traceback


# Registry of running threads
_running_jobs = {}
_lock = threading.Lock()


def spawn_job(job_id, system_type, parameters, dispatch_fn, progress_hook=None):
    """Start a background job. The dispatch_fn runs the actual work.

    If progress_hook is provided, it's called with (current, total)
    as the job progresses.
    """
    def run():
        from app.db import update_job_status, complete_job, fail_job
        try:
            update_job_status(job_id, "running")
            t0 = time.perf_counter()
            result = dispatch_fn(system_type, parameters)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            complete_job(job_id, {
                "result": result,
                "compute_time_ms": elapsed_ms,
            })
        except Exception as e:
            fail_job(job_id, f"{type(e).__name__}: {e}")
        finally:
            with _lock:
                _running_jobs.pop(job_id, None)

    with _lock:
        _running_jobs[job_id] = True

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return job_id


def is_running(job_id):
    with _lock:
        return job_id in _running_jobs


def running_count():
    with _lock:
        return len(_running_jobs)
