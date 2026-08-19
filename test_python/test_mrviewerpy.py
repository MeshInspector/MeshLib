import faulthandler
import os
import platform
import sys
import time
from contextlib import contextmanager

import pytest
from helper import *

mrviewerpy = pytest.importorskip(
    "meshlib.mrviewerpy", reason="mrviewerpy is not available in this build"
)

# On Linux the viewer needs a display server; CI runs the pytest step under `xvfb-run -a`.
# macOS/Windows obtain a context natively (or fall back to NoWindow), so they never skip here.
pytestmark = pytest.mark.skipif(
    platform.system() == "Linux"
    and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
    reason="no DISPLAY/WAYLAND_DISPLAY, run the tests under `xvfb-run -a`",
)

# Generous: the first round-trip also covers viewer construction and GL init on a cold runner.
LAUNCH_TIMEOUT_SEC = 180
CALL_TIMEOUT_SEC = 60


@contextmanager
def bounded(what, seconds=CALL_TIMEOUT_SEC):
    """Bound blocking `CommandLoop` round-trips so a hang fails fast and attributably.

    Those calls hold the GIL while they wait, so no Python-level timer can ever fire -
    `faulthandler`'s watchdog runs outside the GIL and is the only stdlib way to turn a
    hang into a dumped traceback instead of letting it burn the whole step timeout.
    """
    print(f"[mrviewerpy] {what} (watchdog {seconds}s)", file=sys.stderr, flush=True)
    faulthandler.dump_traceback_later(seconds, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture(scope="module")
def viewer():
    """Launch the viewer once for the whole module.

    `launchDefaultViewer` has a process-wide first-launch guard and all of `test_python`
    runs in a single pytest process, so the viewer can be launched exactly once - never
    per test - and everything needing it must live in this module.
    """
    params = mrviewerpy.ViewerLaunchParams()
    # `windowMode` defaults to HideInit, which needs a real window; TryHidden degrades to
    # NoWindow when no GL context is obtainable, which still runs the command loop.
    params.windowMode = mrviewerpy.ViewerLaunchParamsMode.TryHidden
    params.name = "MeshLib test_mrviewerpy"
    mrviewerpy.launch(params, mrviewerpy.ViewerSetup())

    v = mrviewerpy.Viewer()
    # `launch()` returns immediately, so this blocking call is issued into a loop that is
    # not up yet: it must queue and complete, not fail. Returning at all proves the loop runs.
    with bounded("first blocking call right after launch()", LAUNCH_TIMEOUT_SEC):
        v.skipFrames(1)

    yield v

    with bounded("clearScene() + shutdown()"):
        mrviewerpy.clearScene()
        v.shutdown()
    # give the detached launch thread a moment to leave the event loop and clean up
    time.sleep(1)


def test_blocking_round_trips(viewer):
    cube = mrmesh.makeCube(mrmesh.Vector3f.diagonal(1), mrmesh.Vector3f.diagonal(-0.5))

    with bounded("addMeshToScene + selectByName + getSelectedMeshes"):
        mrviewerpy.addMeshToScene(cube, "cube")
        mrviewerpy.selectByName("cube")
        selected = mrviewerpy.getSelectedMeshes()
    assert len(selected) == 1
    assert selected[0].topology.numValidVerts() == cube.topology.numValidVerts()
    assert selected[0].topology.numValidFaces() == cube.topology.numValidFaces()

    with bounded("preciseFitDataViewport"):
        viewer.preciseFitDataViewport()

    # the loop must keep servicing commands, not just the first one
    with bounded("repeated blocking calls"):
        for _ in range(5):
            viewer.skipFrames(1)
            mrviewerpy.unselectAll()
            mrviewerpy.selectByName("cube")
        assert len(mrviewerpy.getSelectedMeshes()) == 1

    with bounded("clearScene"):
        mrviewerpy.clearScene()
        mrviewerpy.selectByType("Meshes")
        assert len(mrviewerpy.getSelectedObjects()) == 0


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="only Linux is guaranteed a real GL context (llvmpipe under Xvfb); headless "
    "Windows GL may be GDI 1.1, where TryHidden degrades to NoWindow and nothing renders",
)
def test_capture_screenshot(viewer, tmp_path):
    with bounded("scene setup"):
        mrviewerpy.clearScene()
        mrviewerpy.addMeshToScene(mrmesh.makeTorus(2, 1, 32, 32, None), "torus")
        mrviewerpy.selectByName("torus")
        viewer.preciseFitDataViewport()
        viewer.skipFrames(2)

    path = tmp_path / "screenshot.png"
    with bounded("captureScreenShot"):
        viewer.captureScreenShot(str(path))

    assert path.exists(), "captureScreenShot() wrote no file"
    assert path.stat().st_size > 0, "captureScreenShot() wrote an empty file"

    with bounded("clearScene"):
        mrviewerpy.clearScene()
