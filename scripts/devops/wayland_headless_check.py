"""Run MeshViewer under a Wayland compositor and read its framebuffer size through MCP.

usage: wayland_headless_check.py <exe> <port> <expected_width> <expected_height> <screenshot.png> [extra viewer args...]
The framebuffer is read twice: right after the MCP server answers (usually before the first frame is presented,
so the compositor has not yet told the client its output scale) and again 4 s later; the second reading is checked.
Exit code 1 when it differs from the expectation or the viewer does not exit cleanly.
"""
import asyncio, json, os, subprocess, sys, time

# MeshViewer's SSE endpoint answers with a doubled `Content-Type: text/event-stream, text/event-stream`;
# httpx2's EventSource compares the header for equality and rejects it, so that check is disabled here.
import httpx2._sse as _sse
for _obj in list(vars(_sse).values()):
    if isinstance(_obj, type) and hasattr(_obj, "_check_content_type"):
        _obj._check_content_type = lambda self: None

from mcp import ClientSession
from mcp.client.sse import sse_client

exe, port = sys.argv[1], int(sys.argv[2])
expected = (int(sys.argv[3]), int(sys.argv[4]))
screenshot = os.path.abspath(sys.argv[5])
extra = sys.argv[6:]
url = f"http://127.0.0.1:{port}/sse"
failures = []
t0 = time.time()


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


async def capture(s, path):
    shot = await s.call_tool("viewer_captureScreenshot", {"includeUi": True, "filePath": path})
    sc = shot.structured_content or {}
    info = await s.call_tool("system_info", {})
    win = (info.structured_content or {}).get("Window Info", {})
    size = (sc.get("width"), sc.get("height"))
    print(f"[{time.time() - t0:.2f} s] window capture {size}, system_info: {json.dumps(win)}", flush=True)
    return size


async def drive():
    async with sse_client(url, timeout=10, sse_read_timeout=120) as (rd, wr):
        async with ClientSession(rd, wr) as s:
            await s.initialize()
            first = await capture(s, screenshot.replace(".png", "-first.png"))
            await asyncio.sleep(4)
            second = await capture(s, screenshot)
            print(f"framebuffer first {first}, after 4 s {second}", flush=True)
            check(second == expected, f"framebuffer size after 4 s {second} == expected {expected}")
            await s.call_tool("viewer_shutdown", {})


def main():
    log = open("viewer_stdout.txt", "wb")
    proc = subprocess.Popen([exe, "-mcpPort", str(port), "-console", *extra], stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 120
    last = None
    while True:
        try:
            asyncio.run(drive())
            break
        except Exception as e:  # the server is not up yet, or it died
            last = e
            if proc.poll() is not None or time.time() > deadline:
                check(False, f"MCP session failed: {type(last).__name__}: {str(last)[:300]}")
                break
            time.sleep(1)
    try:
        code = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        code = proc.wait()
        check(False, "viewer did not exit within 60 s after viewer_shutdown")
    check(code == 0, f"viewer exit code {code}")
    log.close()
    text = open("viewer_stdout.txt", "rb").read().decode("utf-8", "replace")
    print("--- viewer log lines of interest ---")
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in ("glfw", "wayland", "resize", "window pos", "scale", "splash", "error", "warn", "log file", "debug")):
            print("  " + line[:300])
    print("--- end ---")
    sys.exit(1 if failures else 0)


main()
