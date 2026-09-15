#include "MRViewer/MRViewer.h"
#include "MRViewer/MRPythonAppendCommand.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRMouseController.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRSetupViewer.h"
#include "MRPython/MRPython.h"
#include "MRMesh/MRViewportId.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRImage.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRMesh/MRStringConvert.h"
#include "MRViewer/MRGladGlfw.h"
#include <pybind11/stl.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#ifdef __APPLE__
#include <pthread.h>
#endif

#pragma message("mrviewerpy pybind internals magic: " PYBIND11_INTERNALS_ID)

MR_INIT_PYTHON_MODULE_PRECALL( mrviewerpy, [] ()
{
    try
    {
        pybind11::module_::import( "meshlib.mrmeshpy" );
    }
    catch ( const pybind11::error_already_set& )
    {
        pybind11::module_::import( "mrmeshpy" );
    }
} )

static void pythonCaptureScreenShot( MR::Viewer* viewer, const char* path )
{
    MR::CommandLoop::runCommandFromGUIThread( [&] ()
    {
        auto image = viewer->captureSceneScreenShot();
        (void)MR::ImageSave::toAnySupportedFormat( image, path ); //TODO: process potential error
    } );
}

static void pythonCaptureUIScreenShot( MR::Viewer* viewer, const char* path )
{
    auto filename = MR::pathFromUtf8( path );
    MR::CommandLoop::runCommandFromGUIThread( [filename, viewer] ()
    {
        viewer->captureUIScreenShot( [filename] ( const MR::Image& image )
        {
            (void)MR::ImageSave::toAnySupportedFormat( image, filename ); //TODO: process potential error
        } );
    } );
}

static void pythonSkipFrames( MR::Viewer* viewer, int frames )
{
    (void)viewer;
    while ( frames > 0 )
    {
        frames--;
        MR::CommandLoop::runCommandFromGUIThread( []{} );
    }
}

static void pythonShowSceneTree( MR::Viewer* viewer, bool show )
{
    if ( !viewer )
        return;
    MR::CommandLoop::runCommandFromGUIThread( [viewer,show]
    {
        if ( auto ribbonMenu = viewer->getMenuPluginAs<MR::RibbonMenu>() )
        {
            auto config = MR::RibbonMenuUIConfig();
            config.topLayout = MR::RibbonTopPanelLayoutMode::None;
            config.drawToolbar = false;
            config.drawScenePanel = show;
            ribbonMenu->setMenuUIConfig( config );
            viewer->incrementForceRedrawFrames( viewer->forceRedrawMinimumIncrementAfterEvents, viewer->swapOnLastPostEventsRedraw );
        }
    } );
}

static void pythonRunLambdaFromGUIThread( pybind11::function func )
{
    // the GUI thread holds no GIL and has no Python thread state of its own, so `func` must be
    // called under `gil_scoped_acquire`; and the caller must drop the GIL for the whole blocking
    // wait, or that acquire can never succeed
    std::optional<pybind11::error_already_set> pyError;
    std::exception_ptr otherError;
    {
        pybind11::gil_scoped_release gilRelease;
        // captured by reference: a blocking command outlives nothing, and a copy of `func` would
        // touch Python reference counts on a thread that holds no GIL
        MR::CommandLoop::runCommandFromGUIThread( [&func, &pyError, &otherError]
        {
            pybind11::gil_scoped_acquire gilAcquire;
            try
            {
                func();
            }
            catch ( pybind11::error_already_set& e )
            {
                // must be stored while the GIL is held here, and rethrown by the caller below,
                // which holds it again; letting it cross as an exception_ptr destroys it on
                // whichever thread drops the last reference
                pyError = std::move( e );
            }
            catch ( ... )
            {
                otherError = std::current_exception();
            }
        } );
    }
    if ( pyError )
        throw std::move( *pyError ); // GIL reacquired, so pybind11 can restore the Python error
    if ( otherError )
        std::rethrow_exception( otherError );
}

namespace
{

using namespace MR;

enum class PythonKeyMod
{
    Empty = 0,
    Ctrl = GLFW_MOD_CONTROL,
    Super = GLFW_MOD_SUPER,
    Shift = GLFW_MOD_SHIFT,
    Alt = GLFW_MOD_ALT,
};
MR_MAKE_FLAG_OPERATORS( PythonKeyMod )

/// this viewer draws no ribbon, and the libraries owning the ribbon items are not loaded here,
/// so reading the schema of their items would only warn that every one of them is not registered
class MinimalRibbonMenu final : public RibbonMenu
{
    void readMenuItemsStructure_() override {}
};

/// viewer setup class for minimal configuration
/// only loads config file (if available) and configures the scene and mouse controls
class MinimalViewerSetup final : public ViewerSetup
{
public:
    void setupBasePlugins( Viewer* viewer ) const override
    {
        auto menu = std::make_shared<MinimalRibbonMenu>();
        menu->setMenuUIConfig( { .topLayout = RibbonTopPanelLayoutMode::None,.drawScenePanel = false,.drawToolbar = false } ); // no scene tree by default
        viewer->setMenuPlugin( menu );
    }
    void setupExtendedLibraries() const override {}
    void unloadExtendedLibraries() const override {}

    void setupConfiguration( Viewer* viewer ) const override
    {
        viewer->resetSettingsFunction = [base = viewer->resetSettingsFunction] ( Viewer* viewer )
        {
            base( viewer );
            resetSettings_( viewer );
        };
        viewer->resetSettingsFunction( viewer );
    }

private:
    static void resetSettings_( Viewer* viewer )
    {
        viewer->glPickRadius = 3;

        auto& mouseController = viewer->mouseController();
        mouseController.setMouseControl( { MouseButton::Right, 0 }, MouseMode::Translation );
        mouseController.setMouseControl( { MouseButton::Middle, 0 }, MouseMode::Rotation );
        mouseController.setMouseControl( { MouseButton::Middle, getGlfwModPrimaryCtrl() }, MouseMode::Roll );
    }
};

// What a launch has reported so far, shared between the thread running the viewer and the one waiting on it
struct LaunchStatus
{
    std::mutex mutex;
    std::condition_variable cv;
    // the viewer reached AfterWindowAppear: the command loop runs and the viewer is drivable
    bool ready{ false };
    // `launchDefaultViewer` returned; with `ready` still false that means it failed
    bool finished{ false };
    int exitCode{ 0 };

    // Queued before the launch, so it is already in the queue when the loop reaches this state; the
    // command runs on the viewer thread once the viewer is up, and is dropped by the
    // `removeCommands( true )` on `Viewer::launch`'s failure path if it never gets there.
    static void queueReadyReport( std::shared_ptr<LaunchStatus> status )
    {
        MR::CommandLoop::appendCommand( [status]
        {
            {
                std::unique_lock lock( status->mutex );
                status->ready = true;
            }
            status->cv.notify_all();
        }, MR::CommandLoop::StartPosition::AfterWindowAppear );
    }

    void reportFinished( int code )
    {
        {
            std::unique_lock lock( mutex );
            finished = true;
            exitCode = code;
        }
        cv.notify_all();
    }

    /// blocks until the viewer is up (true) or the launch has failed (false)
    bool waitReady()
    {
        std::unique_lock lock( mutex );
        cv.wait( lock, [this] { return ready || finished; } );
        return ready;
    }
};

[[noreturn]] void throwLaunchFailed( int exitCode )
{
    // The usual cause by far is a headless session, where `launchInit_` logs "glfwInit failed" and
    // returns non-zero. `launchDefaultViewer` also returns 1 for a second `launch()` call.
    throw std::runtime_error(
        "Viewer could not start: glfwInit failed (no display available?), or the viewer was already "
        "launched once in this process; exit code " + std::to_string( exitCode ) );
}

#ifndef __APPLE__ // the form AppKit forbids, see `launchWithScript`
// The viewer on a detached thread; the caller continues and drives it with blocking calls.
void launchDetached( const MR::Viewer::LaunchParams& params, const MinimalViewerSetup& setup )
{
    // held by shared_ptr: it must outlive this frame if that threw while the detached thread was still running
    auto status = std::make_shared<LaunchStatus>();
    LaunchStatus::queueReadyReport( status );

    std::thread launchThread { [params, setup, status]
    {
        MR::SetCurrentThreadName( "PythonAppLaunchThread" );
        status->reportFinished( MR::launchDefaultViewer( params, setup ) );
    } };
    launchThread.detach();

    // Wait for the launch to report one way or the other, instead of returning unconditionally:
    // `launchDefaultViewer` calls `setMainThreadId` from the new thread, so returning early leaves a
    // blocking call right after `launch()` with no command loop, and a failed launch used to be
    // indistinguishable from a successful one.
    pybind11::gil_scoped_release gilRelease; // the launch thread must not be blocked by the GIL here
    if ( !status->waitReady() )
        throwLaunchFailed( status->exitCode ); // `finished` alone; both may be set if params.startEventLoop is false
}
#endif

// The viewer on the calling thread and `script` on a worker thread. The only way on macOS, where
// AppKit runs a GUI on the process main thread and nowhere else: GLFW traps inside `glfwInit` on
// any other thread and takes the interpreter down with it (SIGTRAP, exit 133), NoWindow mode included.
void launchWithScript( const MR::Viewer::LaunchParams& params, const MinimalViewerSetup& setup, const pybind11::object& script )
{
#ifdef __APPLE__
    if ( !pthread_main_np() )
        throw std::runtime_error( "On macOS launch( script ) must be called from the main thread" );
#endif
    auto status = std::make_shared<LaunchStatus>();
    LaunchStatus::queueReadyReport( status );

    // the script's exception, re-raised on this thread once it holds the GIL again
    std::optional<pybind11::error_already_set> scriptError;
    std::exception_ptr otherError;
    std::thread scriptThread { [&]
    {
        MR::SetCurrentThreadName( "PythonScriptThread" );
        if ( !status->waitReady() )
            return; // the launch failed, and the calling thread reports it
        {
            pybind11::gil_scoped_acquire gilAcquire;
            try
            {
                script();
            }
            catch ( pybind11::error_already_set& e )
            {
                scriptError = std::move( e ); // stored under the GIL, see `pythonRunLambdaFromGUIThread`
            }
            catch ( ... )
            {
                otherError = std::current_exception();
            }
        }
        // The script is over, so is the viewer, like the interpreter exit that ends a detached launch.
        // Dropped silently if the loop has already closed, e.g. because the user closed the window.
        MR::CommandLoop::appendCommand( [] { MR::getViewerInstance().stopEventLoop(); } );
        MR::getViewerInstance().postEmptyEvent();
    } };

    {
        pybind11::gil_scoped_release gilRelease; // the script thread runs Python while this one runs the viewer
        status->reportFinished( MR::launchDefaultViewer( params, setup ) );
        // a script still running when the window is closed finishes first: its next viewer call raises
        scriptThread.join();
    }
    if ( scriptError )
        throw std::move( *scriptError ); // GIL reacquired, so pybind11 can restore the Python error
    if ( otherError )
        std::rethrow_exception( otherError );
    if ( !status->ready )
        throwLaunchFailed( status->exitCode );
}

void pythonLaunch( const MR::Viewer::LaunchParams& params, const MinimalViewerSetup& setup, const pybind11::object& script )
{
    if ( !script.is_none() )
    {
        if ( !PyCallable_Check( script.ptr() ) )
            throw std::invalid_argument( "launch(): `script` must be callable" );
        launchWithScript( params, setup, script );
        return;
    }
#ifdef __APPLE__
    throw std::runtime_error( "On macOS launch() needs a `script` callable: AppKit runs the viewer on the main thread only, "
        "so launch( script=main ) runs the viewer here and `main` on a worker thread, and returns when `main` does" );
#else
    launchDetached( params, setup );
#endif
}

} // namespace

MR_ADD_PYTHON_CUSTOM_DEF( mrviewerpy, Viewer, [] ( pybind11::module_& m )
{
    pybind11::class_<MinimalViewerSetup>( m, "ViewerSetup" ).
        def( pybind11::init<>() );

    pybind11::enum_<MR::Viewer::LaunchParams::WindowMode>( m, "ViewerLaunchParamsMode" ).
        value( "Hide", MR::Viewer::LaunchParams::WindowMode::Hide, "Don't show window" ).
        value( "HideInit", MR::Viewer::LaunchParams::WindowMode::HideInit, "Show window after init" ).
        value( "Show", MR::Viewer::LaunchParams::WindowMode::Show, "Show window immediately" ).
        value( "TryHidden", MR::Viewer::LaunchParams::WindowMode::TryHidden, "Launches in \"Hide\" mode if OpenGL is present and \"NoWindow\" if it is not" ).
        value( "NoWindow", MR::Viewer::LaunchParams::WindowMode::NoWindow, "Don't initialize GL window (don't call GL functions)(force `isAnimating`)" );

    pybind11::enum_<MR::MouseButton>( m, "MouseButton" )
        .value( "Left", MR::MouseButton::Left )
        .value( "Right", MR::MouseButton::Right )
        .value( "Middle", MR::MouseButton::Middle )
    ;

    pybind11::enum_<PythonKeyMod>( m, "KeyMod" )
        .value( "Empty", PythonKeyMod::Empty )
        .value( "Ctrl", PythonKeyMod::Ctrl )
        .value( "Super", PythonKeyMod::Super )
        .value( "Shift", PythonKeyMod::Shift )
        .value( "Alt", PythonKeyMod::Alt )
        .def( pybind11::self | pybind11::self )
        .def( pybind11::self & pybind11::self )
        .def( ~pybind11::self )
    ;

    pybind11::class_<MR::Viewer::LaunchParams>( m, "ViewerLaunchParams", "This struct contains rules for viewer launch" ).
        def( pybind11::init<>() ).
        def_readwrite( "animationMaxFps", &MR::Viewer::LaunchParams::animationMaxFps, "max fps if animating" ).
        def_readwrite( "fullscreen", &MR::Viewer::LaunchParams::fullscreen, "if true starts fullscreen" ).
        def_readwrite( "width", &MR::Viewer::LaunchParams::width ).
        def_readwrite( "height", &MR::Viewer::LaunchParams::height ).
        def_readwrite( "isAnimating", &MR::Viewer::LaunchParams::isAnimating, "if true - calls render without system events" ).
        def_readwrite( "name", &MR::Viewer::LaunchParams::name, "Window name" ).
        def_readwrite( "windowMode", &MR::Viewer::LaunchParams::windowMode );

    pybind11::class_<MR::Viewport>( m, "Viewport",
        "Viewport is a rectangular area, in which the objects of interest are going to be rendered.\n"
        "An application can have a number of viewports each with its own ID." ).
        def( "cameraLookAlong", MR::pythonRunFromGUIThread( &MR::Viewport::cameraLookAlong ),
            pybind11::arg( "dir" ), pybind11::arg( "up" ),
            "Set camera look direction and up direction (they should be perpendicular)\n"
            "this function changes camera position and do not change camera spot (0,0,0) by default\n"
            "to change camera position use setCameraTranslation after this function" ).
        def( "cameraRotateAround", MR::pythonRunFromGUIThread( &MR::Viewport::cameraRotateAround ),
            pybind11::arg( "axis" ), pybind11::arg( "angle" ),
            "Rotates camera around axis +direction applied to axis point\n"
            "note: this can make camera clip objects (as far as distance to scene center is not fixed)" ).
        def( "projectToViewportSpace", []( const MR::Viewport& v, const MR::Vector3f& input )
            {
                MR::Vector3f ret;
                MR::pythonAppendOrRun( [&]{ ret = v.projectToViewportSpace( input ); } );
                return ret;
            }, "Project world space point to viewport coordinates (in pixels), (0,0) will be at the top-left corner of the viewport." ).
        def_readonly( "id", &MR::Viewport::id )
    ;

    pybind11::enum_<MR::FitMode>( m, "ViewportFitMode", "Fit mode ( types of objects for which the fit is applied )" ).
        value( "Visible", MR::FitMode::Visible, "fit all visible objects" ).
        value( "SelectedObjects", MR::FitMode::SelectedObjects, "fit only selected objects" ).
        value( "SelectedPrimitives", MR::FitMode::SelectedPrimitives, "fit only selected primitives" );

    pybind11::class_<MR::FitDataParams>( m, "ViewportFitDataParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "factor", &MR::FitDataParams::factor, "part of the screen for scene location" ).
        def_readwrite( "snapView", &MR::FitDataParams::snapView, "snapView - to snap camera angle to closest canonical quaternion" ).
        def_readwrite( "mode", &MR::FitDataParams::mode, "fit mode" );

    pybind11::class_<MR::Viewer, std::unique_ptr<MR::Viewer, pybind11::nodelete>>( m, "Viewer", "GLFW-based mesh viewer" ).
        def( pybind11::init( [] ()
    {
        return std::unique_ptr<MR::Viewer, pybind11::nodelete>( MR::Viewer::instance() );
    } ) ).
        def( "viewport", ( MR::Viewport& ( MR::Viewer::* )( MR::ViewportId ) )& MR::Viewer::viewport,
            pybind11::arg_v( "viewportId", MR::ViewportId(), "meshlib.mrmeshpy.ViewportId()" ), pybind11::return_value_policy::reference_internal,
            "Return the current viewport, or the viewport corresponding to a given unique identifier\n"
            "\tviewportId - unique identifier corresponding to the desired viewport (current viewport if 0)" ).
        def( "incrementForceRedrawFrames", MR::pythonRunFromGUIThread( &MR::Viewer::incrementForceRedrawFrames ),
            pybind11::arg( "num" ) = 1,
            pybind11::arg( "swapOnLastOnly" ) = false,
            "Increment number of forced frames to redraw in event loop\n"
            "if `swapOnLastOnly` only last forced frame will be present on screen and all previous will not" ).
        def( "skipFrames", pythonSkipFrames, pybind11::arg("frames") ).
        def( "preciseFitDataViewport", MR::pythonRunFromGUIThread( (void(MR::Viewer::*)( MR::ViewportMask, const MR::FitDataParams& )) &MR::Viewer::preciseFitDataViewport ),
            pybind11::arg_v( "vpList", MR::ViewportMask::all(), "meshlib.mrmeshpy.ViewportMask.all()" ),
            pybind11::arg_v( "params", MR::FitDataParams(), "ViewportFitDataParams()" ),
            "Calls fitData and change FOV to match the screen size then\n"
            "params - params fit data" ).
        def( "captureScreenShot", &pythonCaptureScreenShot,pybind11::arg("path"),
            "Captures part of window (redraw 3d scene over UI (without redrawing UI))" ).
        def( "captureUIScreenShot", &pythonCaptureUIScreenShot, pybind11::arg( "path" ),
            "Captures full window screenshot with UI" ).
        def( "shutdown", MR::pythonRunFromGUIThread( &MR::Viewer::stopEventLoop ), "sets stop event loop flag (this flag is glfwShouldWindowClose equivalent)" ).
        // Input events:
        def( "mouseDown",
            []( MR::Viewer& v, MR::MouseButton b, PythonKeyMod m )
            {
                v.emplaceEvent( "simulatedMouseDown", [&v, b, m]{
                    v.mouseDown( b, int( m ) );
                } );
            },
            pybind11::arg( "button" ), pybind11::arg_v( "modifier", PythonKeyMod{}, "meshlib.mrviewerpy.KeyMod.Empty" ), "Simulate mouse down event."
        ).
        def( "mouseUp",
            []( MR::Viewer& v, MR::MouseButton b, PythonKeyMod m )
            {
                v.emplaceEvent( "simulatedMouseUp", [&v, b, m]{
                    v.mouseUp( b, int( m ) );
                } );
            },
            pybind11::arg( "button" ), pybind11::arg_v( "modifier", PythonKeyMod{}, "meshlib.mrviewerpy.KeyMod.Empty" ), "Simulate mouse up event."
        ).
        def( "mouseMove",
            []( MR::Viewer& viewer, int x, int y )
            {
                MR::pythonAppendOrRun( [&viewer, x, y]
                {
                    glfwSetCursorPos( viewer.window, double( x ) / viewer.pixelRatio, double( y ) / viewer.pixelRatio );

                    // On Windows `glfwSetCursorPos()` automatically sends the `mouseMove()` event. On Linux it doesn't, so we need this:
                    auto eventCall = [&viewer, x, y]{ viewer.mouseMove( x, y ); };
                    viewer.emplaceEvent( "simulatedMouseMove", eventCall, false );
                } );
            },
            pybind11::arg( "x" ), pybind11::arg( "y" ),
            "Simulate mouse move event.\n"
            "NOTE: Some plugins need at least TWO `mouseMove()`s in a row (possibly with the same position). If you're having issues, try sending two events."
        ).
        def( "getMousePos",
            []( const MR::Viewer& )
            {
                double x = -1, y = -1;
                MR::pythonAppendOrRun( [&x, &y]
                {
                    const MR::Viewer &v = MR::getViewerInstance();
                    if ( v.window )
                    {
                        glfwGetCursorPos( v.window, &x, &y );
                        x *= v.pixelRatio;
                        y *= v.pixelRatio;
                    }
                } );
                return MR::Vector2f( float( x ), float( y ) );
            },
            "Get the current mouse position."
        ).
        // Coord projections:
        def( "viewportToScreen", &MR::Viewer::viewportToScreen, "Convert viewport coordinates to to screen coordinates" ).
        def( "showSceneTree", &pythonShowSceneTree, pybind11::arg( "show" ), "Shows or hide scene tree" );

    m.def( "launch", &pythonLaunch,
        pybind11::arg_v( "params", MR::Viewer::LaunchParams(), "ViewerLaunchParams()" ),
        pybind11::arg_v( "setup", MinimalViewerSetup(), "ViewerSetup()" ),
        pybind11::arg_v( "script", pybind11::none(), "None" ),
        "Starts default viewer with given params and setup.\n"
        "Without `script`: the viewer runs on a background thread, and this returns once it is up and can accept calls.\n"
        "With `script`, a callable taking no arguments: the viewer runs on the calling thread and `script` on a worker thread, "
        "which drives it with the same calls; this returns once `script` has returned, closing the viewer, "
        "or once the window was closed and `script` has finished. An exception raised by `script` is re-raised here.\n"
        "On macOS `script` is required and this must be called from the main thread, "
        "because macOS runs a GUI on the process main thread only.\n"
        "Raises RuntimeError if the viewer could not start - with no display available, for instance - "
        "or if it was already launched once in this process." );

    m.def( "runFromGUIThread", &pythonRunLambdaFromGUIThread, pybind11::arg( "lambda" ), "Executes given function from GUI thread, and returns after it is done" );
} )
