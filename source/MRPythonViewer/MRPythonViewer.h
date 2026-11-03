#pragma once

// The API of the `mrviewerpy` Python module, generated from this header over the C bindings (`scripts/mrbind/generate.mk`).
// The calls wait for the viewer's GUI thread, so the module releases the GIL around every call (`--release-gil`).

#include "MRMesh/MRFlagOperators.h"
#include "MRMesh/MRLine.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRViewportId.h"

#include <functional>
#include <string>

namespace MR::PythonViewer
{

/// viewer setup class for minimal configuration
/// only loads config file (if available) and configures the scene and mouse controls
struct ViewerSetup
{
};

enum class ViewerLaunchParamsMode
{
    Show, ///< Show window immediately
    HideInit, ///< Show window after init
    Hide, ///< Don't show window
    TryHidden, ///< Launches in "Hide" mode if OpenGL is present and "NoWindow" if it is not
    NoWindow, ///< Don't initialize GL window (don't call GL functions)(force `isAnimating`)
};

/// This struct contains rules for viewer launch
struct ViewerLaunchParams
{
    int animationMaxFps = 30; ///< max fps if animating
    bool fullscreen = false; ///< if true starts fullscreen
    int width = 0;
    int height = 0;
    bool isAnimating = false; ///< if true - calls render without system events
    std::string name = "MRViewer"; ///< Window name
    ViewerLaunchParamsMode windowMode = ViewerLaunchParamsMode::HideInit;
};

enum class MouseButton
{
    Left,
    Right,
    Middle,
};

enum class KeyMod
{
    Empty = 0,
    Ctrl = 0x0002,
    Super = 0x0008,
    Shift = 0x0001,
    Alt = 0x0004,
};
MR_MAKE_FLAG_OPERATORS( KeyMod )

/// Returns the main control modifier of this platform: `KeyMod.Super` (Cmd) on macOS, `KeyMod.Ctrl` otherwise.
[[nodiscard]] KeyMod getPrimaryCtrlKeyMod();

/// Fit mode ( types of objects for which the fit is applied )
enum class ViewportFitMode
{
    Visible, ///< fit all visible objects
    SelectedPrimitives, ///< fit only selected primitives
    SelectedObjects, ///< fit only selected objects
};

struct ViewportFitDataParams
{
    float factor = 1.f; ///< part of the screen for scene location
    bool snapView = false; ///< snapView - to snap camera angle to closest canonical quaternion
    ViewportFitMode mode = ViewportFitMode::Visible; ///< fit mode
};

/// Viewport is a rectangular area, in which the objects of interest are going to be rendered.
/// An application can have a number of viewports each with its own ID.
class Viewport
{
public:
    explicit Viewport( ViewportId id ) : id( id ) {}

    const ViewportId id;

    /// Set camera look direction and up direction (they should be perpendicular)
    /// this function changes camera position and do not change camera spot (0,0,0) by default
    /// to change camera position use setCameraTranslation after this function
    void cameraLookAlong( const Vector3f& dir, const Vector3f& up ) const;

    /// Rotates camera around axis +direction applied to axis point
    /// note: this can make camera clip objects (as far as distance to scene center is not fixed)
    void cameraRotateAround( const Line3f& axis, float angle ) const;

    /// Project world space point to viewport coordinates (in pixels), (0,0) will be at the top-left corner of the viewport.
    [[nodiscard]] Vector3f projectToViewportSpace( const Vector3f& input ) const;
};

/// GLFW-based mesh viewer
class Viewer
{
public:
    /// Return the current viewport, or the viewport corresponding to a given unique identifier
    /// 	viewportId - unique identifier corresponding to the desired viewport (current viewport if 0)
    [[nodiscard]] Viewport viewport( ViewportId viewportId = {} ) const;

    /// Increment number of forced frames to redraw in event loop
    /// if `swapOnLastOnly` only last forced frame will be present on screen and all previous will not
    void incrementForceRedrawFrames( int num = 1, bool swapOnLastOnly = false ) const;

    void skipFrames( int frames ) const;

    /// Calls fitData and change FOV to match the screen size then
    /// params - params fit data
    void preciseFitDataViewport( ViewportMask vpList = ViewportMask::all(), const ViewportFitDataParams& params = {} ) const;

    /// Captures part of window (redraw 3d scene over UI (without redrawing UI))
    void captureScreenShot( const std::string& path ) const;

    /// Captures full window screenshot with UI
    void captureUIScreenShot( const std::string& path ) const;

    /// sets stop event loop flag (this flag is glfwShouldWindowClose equivalent); the viewer's launch then returns exitCode; interruptible = False: no interruptCloseSignal slot can cancel the stop
    void shutdown( int exitCode = 0, bool interruptible = true ) const;

    /// Simulate mouse down event.
    void mouseDown( MouseButton button, KeyMod modifier = KeyMod::Empty ) const;

    /// Simulate mouse up event.
    void mouseUp( MouseButton button, KeyMod modifier = KeyMod::Empty ) const;

    /// Simulate mouse move event.
    /// NOTE: Some plugins need at least TWO `mouseMove()`s in a row (possibly with the same position). If you're having issues, try sending two events.
    void mouseMove( int x, int y ) const;

    /// Get the current mouse position.
    [[nodiscard]] Vector2f getMousePos() const;

    /// Convert viewport coordinates to to screen coordinates
    [[nodiscard]] Vector3f viewportToScreen( const Vector3f& viewportPoint, ViewportId id ) const;

    /// Shows or hide scene tree
    void showSceneTree( bool show ) const;
};

/// Starts the viewer with the given params and setup, and returns once it is up and accepts calls.
/// On Windows and Linux the window opens here, and the script keeps running alongside it.
/// On macOS the viewer runs on the calling thread, which must be the main one: the window opens in showViewer(), or at the interactive prompt, where it stays live while the interpreter waits for input, in input() too.
/// Raises RuntimeError if the viewer could not start (no display, or already launched once in this process).
void launch( const ViewerLaunchParams& params = {}, const ViewerSetup& setup = {} );

/// Keeps the window open until the user closes it or shutdown() is called from another thread.
/// On macOS the window opens here, so this call is required there; make it from the main thread.
/// After it returns the viewer is closed for good, and further viewer calls raise RuntimeError.
void showViewer();

/// Executes given function from GUI thread, and returns after it is done
void runFromGUIThread( const std::function<void()>& lambda );

/// Registered with `atexit` by the module, and removed from it: closes the viewer the way a closed window would, if it's still up.
void shutViewerAtExit();

} // namespace MR::PythonViewer
