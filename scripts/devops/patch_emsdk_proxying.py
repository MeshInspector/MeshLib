#!/usr/bin/env python3
"""Apply emscripten-core/emscripten#26582 to an emsdk that predates 5.0.5.

`emscripten_proxy_finish` signals the proxy context's condition variable *after*
releasing its mutex. In that window the waiting thread can take the mutex, see the
state as DONE, skip `pthread_cond_wait`, return from `emscripten_proxy_sync_with_ctx`
and destroy the context -- which lives on its own stack -- so the signal lands on a
condvar that no longer exists and the thread that should have woken waits forever.

Under `-pthread` every syscall is proxied to the main thread, so any worker thread
doing file I/O can hang. Upstream swapped the two statements; that is all this does.

Idempotent, and loud if the source ever stops matching: emsdk 5.0.5 and newer already
carry the fix, and this script is meant to disappear when we get there.
"""
import io
import sys

PATH = "/emsdk/upstream/emscripten/system/lib/pthread/proxying.c"

# (state assigned before the signal, comment to leave behind)
SITES = [
    ("DONE",
     "    // Signal must come before unlock, or the waiter can see DONE and free the\n"
     "    // ctx before we unlock. https://github.com/emscripten-core/emscripten/pull/26582\n"),
    ("CANCELED",
     "    // Signal must be first, see the comment in emscripten_proxy_finish.\n"),
]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else PATH
    s = io.open(path, encoding="utf-8").read()

    already = 0
    for state, comment in SITES:
        unlock_first = (
            "    ctx->sync.state = %s;\n" % state
            + ("    remove_active_ctx(ctx);\n" if state == "DONE" else "")
            + "    pthread_mutex_unlock(&ctx->sync.mutex);\n"
            "    pthread_cond_signal(&ctx->sync.cond);\n"
        )
        signal_first = (
            "    ctx->sync.state = %s;\n" % state
            + ("    remove_active_ctx(ctx);\n" if state == "DONE" else "")
            + comment
            + "    pthread_cond_signal(&ctx->sync.cond);\n"
            "    pthread_mutex_unlock(&ctx->sync.mutex);\n"
        )
        if signal_first in s:
            already += 1
            continue
        if unlock_first not in s:
            sys.stderr.write(
                "%s: no unpatched '%s' site found in %s -- refusing to guess. If the emsdk "
                "version moved past 5.0.5 this patch is redundant; drop it.\n"
                % (sys.argv[0], state, path))
            return 1
        s = s.replace(unlock_first, signal_first, 1)

    if already == len(SITES):
        print("proxying.c already carries emscripten#26582, nothing to do")
        return 0

    io.open(path, "w", encoding="utf-8").write(s)
    print("patched %s with emscripten#26582 (%d site(s))" % (path, len(SITES) - already))
    return 0


if __name__ == "__main__":
    sys.exit(main())
