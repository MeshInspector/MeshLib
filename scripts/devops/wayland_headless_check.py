"""Run MeshViewer under a Wayland compositor and read its framebuffer size through MCP.

usage: wayland_headless_check.py <exe> <port> <expected_width> <expected_height> <screenshot.png> [extra viewer args...]
Exit code 1 when the framebuffer size differs from the expectation or the viewer does not exit cleanly.
"""
import asyncio, json, os, subprocess, sys, time

from mcp import ClientSession
from mcp.client.sse import sse_client

exe, port = sys.argv[1], int(sys.argv[2])
expected = (int(sys.argv[3]), int(sys.argv[4]))
screenshot = os.path.abspath(sys.argv[5])
extra = sys.argv[6:]
url = f"http://127.0.0.1:{port}/sse"
failures = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


async def drive():
    async with sse_client(url, timeout=10, sse_read_timeout=120) as (rd, wr):
        async with ClientSession(rd, wr) as s:
            await s.initialize()
            info = await s.call_tool("system_info", {})
            print("system_info:", json.dumps(info.structured_content)[:1500], flush=True)
            shot = await s.call_tool("viewer_captureScreenshot", {"includeUi": True, "filePath": screenshot})
            sc = shot.structured_content or {}
            size = (sc.get("width"), sc.get("height"))
            print("framebuffer (window capture):", size, flush=True)
            check(size == expected, f"framebuffer size {size} == expected {expected}")
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
        if any(k in low for k in ("glfw", "wayland", "resize", "window pos", "scale", "splash", "error", "warn", "log file")):
            print("  " + line[:300])
    print("--- end ---")
    sys.exit(1 if failures else 0)


main()
