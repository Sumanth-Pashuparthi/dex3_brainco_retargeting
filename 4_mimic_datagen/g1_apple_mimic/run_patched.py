"""Run an Arena script with the Lightwheel asset SDK made tolerant of a slow API.

Arena's ``object_library.py`` calls ``lightwheel_sdk`` at import time for every Lightwheel object
(microwave, broccoli, ...) whether or not the task uses them, and ``lightwheel_sdk.client`` ships a
hard-coded 10 s read timeout with a single retry. ``api-dev.lightwheel.net`` answers the same request
in anywhere from 1.5 s to 25 s, so every Isaac Sim process (annotation, each generation worker) has
a real chance of dying at start-up with ``Read timed out``. The SDK exposes neither the timeout nor
the retry count through the environment.

This wrapper raises the timeout on the SDK's module-level client and retries with back-off, then
executes the target script as ``__main__`` with the remaining arguments untouched::

    /isaac-sim/python.sh g1_apple_mimic/run_patched.py \
        isaaclab_arena/scripts/imitation_learning/annotate_demos.py --headless ...

Environment: ``LW_TIMEOUT_S`` (default 90), ``LW_RETRIES`` (default 6).
"""

from __future__ import annotations

import os
import runpy
import sys
import time


def patch_lightwheel() -> None:
    try:
        import lightwheel_sdk.client as lwc
    except Exception as e:  # noqa: BLE001
        print(f"[run_patched] lightwheel_sdk not importable ({e}); running unpatched", flush=True)
        return
    client = getattr(lwc, "lw_client", None)
    if client is None:
        print("[run_patched] lightwheel_sdk.client.lw_client not found; running unpatched", flush=True)
        return
    timeout_s = int(os.environ.get("LW_TIMEOUT_S", "90"))
    retries = int(os.environ.get("LW_RETRIES", "6"))
    client.base_timeout = timeout_s
    orig_post = client.post

    def post(path, **kw):
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return orig_post(path, **kw)
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"[run_patched] lightwheel POST {path} attempt {attempt}/{retries} failed: {e}", flush=True)
                time.sleep(min(30, 3 * attempt))
        raise last  # type: ignore[misc]

    client.post = post
    print(f"[run_patched] lightwheel client: timeout {timeout_s}s, {retries} retries", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    patch_lightwheel()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
