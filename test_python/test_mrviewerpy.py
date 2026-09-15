import faulthandler
import os
import pathlib
import platform
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import pytest
from helper import *

# Imported by the `viewer` fixture, never at module scope, so nothing loads libMRViewer on the
# platforms that skip below.
mrviewerpy = None


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
# how long before the `faulthandler` watchdog kills us the sidecar samples the thread table
WATCHDOG_LEAD_SEC = 10

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


# Sidecar watchdog, run as a child process. `faulthandler` can name the Python frame a hang
# is stuck on, but not why the viewer thread is not answering - and nothing in-process can look,
# because the blocking call owns the GIL. A separate process can: it samples our whole thread
# table twice, so the two shapes a stuck round-trip can have are told apart on the first CI hit.
#  - viewer thread parked (S) with no CPU accruing while the command sits in the queue: the
#    wakeup was lost, i.e. `postEmptyEvent()` never got the loop out of `glfwWaitEvents()`.
#  - viewer or `llvmpipe-*` threads burning CPU (R, cpu > 0): GL work that has not finished,
#    e.g. `captureSceneScreenShot`'s offscreen render plus `glGetTexImage` on llvmpipe.
# `wchan` needs ptrace permission we may not have; state and CPU alone separate the two.
_THREAD_DUMP_SRC = r"""
import os
import sys
import time

pid, delay, caller_tid, what = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
tasks = "/proc/%d/task" % pid


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as e:
        return "<%s>" % type(e).__name__


def snapshot():
    out = {}
    try:
        tids = sorted(os.listdir(tasks), key=int)
    except OSError:
        return out
    for tid in tids:
        # `comm` may hold spaces and parens, so the numbered fields start after the last ')'
        fields = read("%s/%s/stat" % (tasks, tid)).rpartition(")")[2].split()
        if len(fields) < 13:
            continue
        out[tid] = (
            read("%s/%s/comm" % (tasks, tid)),
            fields[0],  # state: R runnable, S sleeping, D uninterruptible
            int(fields[11]) + int(fields[12]),  # utime + stime, in clock ticks
            read("%s/%s/wchan" % (tasks, tid)),
        )
    return out


# Fire before `faulthandler`'s own exit=True kill, and stay quiet if the call returned (pytest
# kills us then) or the parent is gone - a dead pid can be reused by an unrelated process.
deadline = time.monotonic() + delay
while time.monotonic() < deadline:
    if os.getppid() != pid:
        sys.exit(0)
    time.sleep(0.25)

first = snapshot()
time.sleep(2.0)
second = snapshot()
if not second:
    sys.exit(0)

ticks = os.sysconf("SC_CLK_TCK")
lines = [
    "",
    "[mrviewerpy] watchdog: %s still blocked after %gs; thread table of pid %d,"
    " cpu = seconds burned over a 2s window" % (what, delay, pid),
    "%8s %-17s %-5s %6s  %s" % ("tid", "comm", "state", "cpu", "wchan"),
]
for tid, (comm, state, cpu, wchan) in second.items():
    was = first.get(tid)
    lines.append(
        "%8s %-17s %-5s %6.2f  %s%s"
        % (
            tid,
            comm,
            state,
            (cpu - was[2]) / ticks if was else float("nan"),
            wchan,
            "   <- caller" if tid == caller_tid else "",
        )
    )
# straight to fd 2, like faulthandler: pytest runs with -s, and nothing buffered survives its kill
os.write(2, ("\n".join(lines) + "\n").encode())
"""


def _start_thread_dump_watchdog(what, delay):
    """Arm the sidecar above; `None` if it cannot run, which never fails a test."""
    if not os.path.isdir("/proc/self/task"):
        return None
    try:
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                _THREAD_DUMP_SRC,
                str(os.getpid()),
                str(delay),
                str(threading.get_native_id()),
                what,
            ]
        )
    except OSError as e:
        print(f"[mrviewerpy] watchdog not started: {e}", file=sys.stderr, flush=True)
        return None


@contextmanager
def bounded(what, seconds=CALL_TIMEOUT_SEC):
    """Bound blocking `CommandLoop` round-trips so a hang fails fast and attributably.

    Those calls hold the GIL while they wait, so no Python-level timer can ever fire -
    `faulthandler`'s watchdog runs outside the GIL and is the only stdlib way to turn a
    hang into a dumped traceback instead of letting it burn the whole step timeout. It
    dumps Python frames only, hence the out-of-process thread sampler alongside it.
    """
    print(f"[mrviewerpy] {what} (watchdog {seconds}s)", file=sys.stderr, flush=True)
    faulthandler.dump_traceback_later(seconds, exit=True)
    watchdog = _start_thread_dump_watchdog(what, max(1.0, seconds - WATCHDOG_LEAD_SEC))
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        if watchdog is not None:
            try:
                watchdog.terminate()
            except OSError:
                pass
            watchdog.wait()


@pytest.fixture(scope="module")
def viewer():
    """Launch the viewer once for the whole module.

    `launchDefaultViewer` has a process-wide first-launch guard and all of `test_python`
    runs in a single pytest process, so the viewer can be launched exactly once - never
    per test - and everything needing it must live in this module.
    """
    global mrviewerpy
    mrviewerpy = pytest.importorskip(
        "meshlib.mrviewerpy", reason="mrviewerpy is not available in this build"
    )
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


def test_run_from_gui_thread(viewer):
    """The callable must actually run, on the GUI thread, and the call must come back."""
    ran_on = []

    with bounded("runFromGUIThread"):
        mrviewerpy.runFromGUIThread(lambda: ran_on.append(threading.get_ident()))

    assert ran_on, "the callable never ran"
    assert ran_on[0] != threading.get_ident(), "the callable ran on the calling thread"


def test_run_from_gui_thread_propagates_exception(viewer):
    """An exception raised inside the callable must reach the caller with its type intact."""

    class Boom(Exception):
        pass

    def raiser():
        raise Boom("raised on the GUI thread")

    with bounded("runFromGUIThread raising"):
        with pytest.raises(Boom, match="raised on the GUI thread"):
            mrviewerpy.runFromGUIThread(raiser)

    # the loop survived it, so the viewer is still usable
    with bounded("skipFrames after the raising callable"):
        viewer.skipFrames(1)


# --- fail fast when no command loop can run the command --------------------------------
#
# Both cases below need a viewer state the module fixture cannot reach - never launched,
# and already shut down - and `launchDefaultViewer` refuses a second launch per process,
# so each gets its own interpreter. The parent bounds the child, so a regression back to
# blocking forever fails the test in seconds instead of burning the whole pytest step.

# generous enough for a cold launch under llvmpipe, small enough that a hang cannot eat
# the step timeout
CHILD_TIMEOUT_SEC = 120

# `skipFrames` is the blocking call throughout: it posts an empty C++ command, so it exercises
# the wait in `addCommand_` and nothing else. `runFromGUIThread` would post a *Python* callable
# instead, and the GUI thread has no Python thread state to call one with - an unrelated crash
# that would say nothing about the loop.
#
# No `launch()` anywhere, so `CommandLoop` never gets a main thread id: nothing will ever
# call `processCommands` and this command cannot run, whoever waits for it.
_NEVER_LAUNCHED_SRC = r"""
import sys

from meshlib import mrviewerpy

try:
    mrviewerpy.Viewer().skipFrames(1)
except RuntimeError as e:
    print("RAISED %s" % e, flush=True)
    sys.exit(0)
print("RETURNED", flush=True)
sys.exit(1)
"""

# `Viewer::launch` drains and closes the queue on its way out, so a command posted after
# `shutdown()` cannot run either. Retried until the detached launch thread gets there: a
# call still serviced by the dying loop is not yet a verdict, an empty return after it is.
_HEADLESS_LAUNCH_SRC = r"""
import os
import pathlib
import sys

import meshlib.mrmeshpy as mrmesh
from meshlib import mrviewerpy

# so a missing font, rather than the missing display, cannot be what fails the launch
mrmesh.SystemPath.overrideDirectory(
    mrmesh.SystemPath.Directory.Resources,
    pathlib.Path(os.environ["MRVIEWERPY_RESOURCES"]),
)
mrmesh.SystemPath.overrideDirectory(
    mrmesh.SystemPath.Directory.Fonts,
    pathlib.Path(os.environ["MRVIEWERPY_FONTS"]),
)

params = mrviewerpy.ViewerLaunchParams()
params.windowMode = mrviewerpy.ViewerLaunchParamsMode.TryHidden
params.name = "MeshLib test_mrviewerpy headless launch"
try:
    mrviewerpy.launch(params, mrviewerpy.ViewerSetup())
except RuntimeError as e:
    print("LAUNCH_RAISED %s" % e, flush=True)
except BaseException as e:
    print("LAUNCH_RAISED_OTHER %r" % e, flush=True)
    sys.exit(4)
else:
    print("LAUNCH_RETURNED", flush=True)
    sys.exit(2)

# the launch failed: a viewer call must now fail too, rather than park on a loop that never runs
viewer = mrviewerpy.Viewer()
try:
    viewer.skipFrames(1)
except RuntimeError as e:
    print("CALL_RAISED %s" % e, flush=True)
    sys.exit(0)
print("CALL_RETURNED", flush=True)
sys.exit(3)
"""


_AFTER_SHUTDOWN_SRC = r"""
import os
import pathlib
import sys
import time

import meshlib.mrmeshpy as mrmesh
from meshlib import mrviewerpy

mrmesh.SystemPath.overrideDirectory(
    mrmesh.SystemPath.Directory.Resources,
    pathlib.Path(os.environ["MRVIEWERPY_RESOURCES"]),
)
mrmesh.SystemPath.overrideDirectory(
    mrmesh.SystemPath.Directory.Fonts,
    pathlib.Path(os.environ["MRVIEWERPY_FONTS"]),
)

params = mrviewerpy.ViewerLaunchParams()
params.windowMode = mrviewerpy.ViewerLaunchParamsMode.TryHidden
params.name = "MeshLib test_mrviewerpy shutdown"
mrviewerpy.launch(params, mrviewerpy.ViewerSetup())
viewer = mrviewerpy.Viewer()
viewer.skipFrames(1)
print("STAGE alive", flush=True)

viewer.shutdown()
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    try:
        viewer.skipFrames(1)
    except RuntimeError as e:
        print("RAISED %s" % e, flush=True)
        sys.exit(0)
    time.sleep(0.25)
print("RETURNED", flush=True)
sys.exit(1)
"""


class _ChildRun:
    def __init__(self, returncode, stdout, stderr, timed_out):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out

    def report(self):
        exit_state = (
            "killed on timeout" if self.timed_out else f"exit {self.returncode}"
        )
        return (
            f"child {exit_state}\n"
            f"--- child stdout ---\n{self.stdout}"
            f"--- child stderr ---\n{self.stderr}"
        )


def _run_in_child(what, source, env_extra=None, timeout=CHILD_TIMEOUT_SEC):
    """Run `source` in a fresh interpreter, killing it if it blocks."""

    def as_text(v):
        # partial output on `TimeoutExpired` comes back undecoded even with text=True
        if v is None:
            return ""
        return v.decode(errors="replace") if isinstance(v, bytes) else v

    env = dict(os.environ)
    # the child gets no conftest, so hand it whatever made the bindings importable here
    env["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys(p for p in sys.path + [os.getcwd()] if p)
    )
    env.update(env_extra or {})

    print(
        f"[mrviewerpy] {what} in a child (timeout {timeout}s)",
        file=sys.stderr,
        flush=True,
    )
    try:
        done = subprocess.run(
            [sys.executable, "-u", "-c", source],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return _ChildRun(None, as_text(e.stdout), as_text(e.stderr), True)
    return _ChildRun(done.returncode, done.stdout, done.stderr, False)


def test_blocking_call_without_launch_raises():
    """A blocking call issued with no viewer launched must fail, not park forever."""
    pytest.importorskip(
        "meshlib.mrviewerpy", reason="mrviewerpy is not available in this build"
    )

    run = _run_in_child(
        "blocking call with no launch()", _NEVER_LAUNCHED_SRC, timeout=60
    )

    assert not run.timed_out, (
        "the blocking call never returned: it is waiting for a command loop that will "
        "never run\n" + run.report()
    )
    assert run.returncode == 0 and "RAISED" in run.stdout, (
        "the blocking call reported success without running the command\n" + run.report()
    )


def test_blocking_call_after_shutdown_raises():
    """A blocking call issued after `shutdown()` must fail, not report a silent no-op."""
    global mrviewerpy
    mrviewerpy = pytest.importorskip(
        "meshlib.mrviewerpy", reason="mrviewerpy is not available in this build"
    )
    _point_at_bundled_resources()

    run = _run_in_child(
        "blocking call after shutdown()",
        _AFTER_SHUTDOWN_SRC,
        env_extra={
            "MRVIEWERPY_RESOURCES": str(mrmesh.SystemPath.getResourcesDirectory()),
            "MRVIEWERPY_FONTS": str(mrmesh.SystemPath.getFontsDirectory()),
        },
    )

    # never got a running viewer: that is this environment, not the behaviour under test
    if "STAGE alive" not in run.stdout:
        pytest.skip("the child could not launch a viewer to shut down\n" + run.report())

    assert not run.timed_out, (
        "the blocking call never returned after shutdown(): it is waiting on a loop that "
        "has already stopped\n" + run.report()
    )
    assert run.returncode == 0 and "RAISED" in run.stdout, (
        "the blocking call returned as if the command had run on a shut down viewer\n"
        + run.report()
    )


# --- a command dropped by `removeCommands` ---------------------------------------------
#
# The two cases above are refused *up front*: the loop had already closed its queue, or was
# never started, so `addCommand_` never queues anything. This one is the third refusal, and
# the only one where the caller is already parked in the wait: the queue was open when the
# command was posted, `shutdown()` then took the loop down under it, and `removeCommands`
# woke the caller with the command dropped unrun. That wake-up is indistinguishable from a
# normal completion at the condition variable - both set `done` - so before the fail-fast
# change the caller simply returned, reporting success for a command that never ran.
#
# The window is a race by nature: after `shutdown()` returns, the loop may still service a
# command or two before it leaves the event loop, and once it has drained the queue further
# commands are refused up front instead. So the child posts with no pause at all to keep a
# command in flight across the drain, and the test hard-asserts what holds either way - the
# call raises rather than returning, and the refusal is permanent - while reporting whether
# the drop window itself was hit.
_DROPPED_ON_SHUTDOWN_SRC = r"""
import os
import pathlib
import sys
import time

import meshlib.mrmeshpy as mrmesh
from meshlib import mrviewerpy

mrmesh.SystemPath.overrideDirectory(
    mrmesh.SystemPath.Directory.Resources,
    pathlib.Path(os.environ["MRVIEWERPY_RESOURCES"]),
)
mrmesh.SystemPath.overrideDirectory(
    mrmesh.SystemPath.Directory.Fonts,
    pathlib.Path(os.environ["MRVIEWERPY_FONTS"]),
)

params = mrviewerpy.ViewerLaunchParams()
params.windowMode = mrviewerpy.ViewerLaunchParamsMode.TryHidden
params.name = "MeshLib test_mrviewerpy dropped"
mrviewerpy.launch(params, mrviewerpy.ViewerSetup())
viewer = mrviewerpy.Viewer()
viewer.skipFrames(1)
print("STAGE alive", flush=True)

viewer.shutdown()

# no sleep in the loop: a command must be queued at the instant `removeCommands` drains it
served = 0
refusal = None
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    try:
        viewer.skipFrames(1)
    except RuntimeError as e:
        refusal = str(e)
        break
    served += 1

print("SERVED %d" % served, flush=True)
if refusal is None:
    print("RETURNED", flush=True)
    sys.exit(1)
print("REFUSED %s" % refusal, flush=True)

# the loop is gone for good, so no later command may be reported as run either
for _ in range(20):
    try:
        viewer.skipFrames(1)
    except RuntimeError:
        continue
    print("SERVED AFTER REFUSAL", flush=True)
    sys.exit(1)
print("PERMANENT", flush=True)
sys.exit(0)
"""

# `removeCommands` puts this in the error of a caller it woke without running its command
DROPPED_MESSAGE = "stopped before the command was executed"


def test_command_dropped_by_shutdown_raises():
    """A command dropped while its caller waits must raise, not return as if it had run."""
    global mrviewerpy
    mrviewerpy = pytest.importorskip(
        "meshlib.mrviewerpy", reason="mrviewerpy is not available in this build"
    )
    _point_at_bundled_resources()

    run = _run_in_child(
        "blocking calls racing shutdown()",
        _DROPPED_ON_SHUTDOWN_SRC,
        env_extra={
            "MRVIEWERPY_RESOURCES": str(mrmesh.SystemPath.getResourcesDirectory()),
            "MRVIEWERPY_FONTS": str(mrmesh.SystemPath.getFontsDirectory()),
        },
    )

    # never got a running viewer: that is this environment, not the behaviour under test
    if "STAGE alive" not in run.stdout:
        pytest.skip("the child could not launch a viewer to shut down\n" + run.report())

    assert not run.timed_out, (
        "a blocking call never returned while the loop was going down\n" + run.report()
    )
    assert "SERVED AFTER REFUSAL" not in run.stdout, (
        "a blocking call was reported as run after an earlier one had already been "
        "refused: the loop is gone, so nothing can have executed it\n" + run.report()
    )
    assert run.returncode == 0 and "REFUSED" in run.stdout, (
        "a blocking call issued while the loop was shutting down returned as if the "
        "command had run\n" + run.report()
    )

    refusal = next(
        line[len("REFUSED ") :]
        for line in run.stdout.splitlines()
        if line.startswith("REFUSED ")
    )
    # Both refusals are correct behaviour; only the first one went through the drop path,
    # so say which was actually exercised rather than implying the wider one.
    if DROPPED_MESSAGE not in refusal:
        print(
            f"[mrviewerpy] the loop had already closed its queue, so the dropped-command "
            f"path was not hit this run; refused with: {refusal}",
            file=sys.stderr,
            flush=True,
        )


# --- a launch that never comes up -------------------------------------------------------
#
# Above, the loop is the thing that is gone. Here it never starts: `launch()` itself fails,
# so the report has to come from `launch()` rather than from the first command after it.


def test_headless_launch_raises_and_survives():
    """`launch()` with no display must raise, and must not take the interpreter with it.

    The child gets an empty `DISPLAY`/`WAYLAND_DISPLAY` while this process keeps its own, so it
    reproduces the headless case from inside an `xvfb-run` test session. It is a child rather than
    an in-process call because a regression here either hangs or dies by signal.
    """
    global mrviewerpy
    mrviewerpy = pytest.importorskip(
        "meshlib.mrviewerpy", reason="mrviewerpy is not available in this build"
    )
    _point_at_bundled_resources()

    run = _run_in_child(
        "launch() with no display",
        _HEADLESS_LAUNCH_SRC,
        env_extra={
            "DISPLAY": "",
            "WAYLAND_DISPLAY": "",
            "MRVIEWERPY_RESOURCES": str(mrmesh.SystemPath.getResourcesDirectory()),
            "MRVIEWERPY_FONTS": str(mrmesh.SystemPath.getFontsDirectory()),
        },
    )

    assert not run.timed_out, (
        "launch() never returned with no display available\n" + run.report()
    )
    # a viewer actually came up: the child was not headless after all, so nothing was tested
    if "LAUNCH_RETURNED" in run.stdout:
        pytest.skip(
            "the child got a display despite an empty DISPLAY/WAYLAND_DISPLAY\n" + run.report()
        )
    assert "LAUNCH_RAISED" in run.stdout, (
        "launch() failed without raising a RuntimeError\n" + run.report()
    )
    # SIGTRAP-by-AppKit is the macOS face of this bug (exit 133); this asserts the shape of the
    # contract everywhere - a failed launch is an exception, never a dead interpreter
    assert run.returncode is not None and run.returncode >= 0, (
        "the interpreter died by signal instead of raising\n" + run.report()
    )


def test_call_after_failed_launch_raises():
    """After a failed `launch()`, the next viewer call must fail promptly, not deadlock."""
    global mrviewerpy
    mrviewerpy = pytest.importorskip(
        "meshlib.mrviewerpy", reason="mrviewerpy is not available in this build"
    )
    _point_at_bundled_resources()

    run = _run_in_child(
        "viewer call after a failed launch()",
        _HEADLESS_LAUNCH_SRC,
        env_extra={
            "DISPLAY": "",
            "WAYLAND_DISPLAY": "",
            "MRVIEWERPY_RESOURCES": str(mrmesh.SystemPath.getResourcesDirectory()),
            "MRVIEWERPY_FONTS": str(mrmesh.SystemPath.getFontsDirectory()),
        },
    )

    if "LAUNCH_RETURNED" in run.stdout:
        pytest.skip(
            "the child got a display despite an empty DISPLAY/WAYLAND_DISPLAY\n" + run.report()
        )
    assert not run.timed_out, (
        "the call after a failed launch() never returned\n" + run.report()
    )
    assert run.returncode == 0 and "CALL_RAISED" in run.stdout, (
        "the call after a failed launch() did not raise\n" + run.report()
    )
