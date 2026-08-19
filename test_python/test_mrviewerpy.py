import faulthandler
import os
import pathlib
import platform
import sys
import time
from contextlib import contextmanager

import pytest
from helper import *

mrviewerpy = pytest.importorskip(
    "meshlib.mrviewerpy", reason="mrviewerpy is not available in this build"
)

# The viewer can only be launched from Python headless on Linux:
#  - macOS: `launch()` runs the viewer on a detached thread, but GLFW/Cocoa insists on
#    owning the main thread, so the first blocking round-trip never returns.
#  - Windows CI has no OpenGL driver at all ("WGL: The driver does not appear to support
#    OpenGL"), so window creation fails for both 4.3 and 3.3 and the viewer gives up.
# Linux under `xvfb-run -a` gets a real GL 4.5 context from llvmpipe, which is enough for
# both the command loop and `captureScreenShot`.
pytestmark = [
    pytest.mark.skipif(
        platform.system() != "Linux",
        reason="the viewer is only launchable from Python headless on Linux: macOS needs "
        "the GUI on the main thread, Windows CI has no OpenGL driver",
    ),
    pytest.mark.skipif(
        not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
        reason="no DISPLAY/WAYLAND_DISPLAY, run the tests under `xvfb-run -a`",
    ),
]

# Generous: the first round-trip also covers viewer construction and GL init on a cold runner.
LAUNCH_TIMEOUT_SEC = 180
CALL_TIMEOUT_SEC = 60

# one of the fonts the ribbon loads during launch; see `_point_at_bundled_resources`
FONT_NAME = "NotoSansCJK-Regular.ttc"


def _point_at_bundled_resources():
    """Make the fonts the viewer loads during `launch()` findable.

    `SystemPath` resolves the resources directory from the *executable*, which is the
    Python interpreter here - so a developer build (binaries under `build/<config>/bin`
    with the fonts copied next to them) ends up looking in e.g. `/usr/bin/fonts` and finds
    nothing. That is not survivable: ImGui hands back a null font and `launch()` dies
    inside `RibbonFontManager::loadAllFonts`. The wheel already works around this from
    `meshlib/__init__.py` (scripts/wheel/init.py), so only the layouts shipping without
    that shim reach the override below.
    """
    if (mrmesh.SystemPath.getFontsDirectory() / FONT_NAME).is_file():
        return

    # wheel keeps the fonts inside the package, a developer build one level up next to the libs
    package_dir = pathlib.Path(mrviewerpy.__file__).resolve().parent
    for candidate in (package_dir, package_dir.parent):
        if (candidate / FONT_NAME).is_file():
            mrmesh.SystemPath.overrideDirectory(
                mrmesh.SystemPath.Directory.Resources, candidate
            )
            mrmesh.SystemPath.overrideDirectory(
                mrmesh.SystemPath.Directory.Fonts, candidate
            )
            return

    pytest.skip(f"{FONT_NAME} not found next to the bindings, launching would crash")


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
    _point_at_bundled_resources()

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
