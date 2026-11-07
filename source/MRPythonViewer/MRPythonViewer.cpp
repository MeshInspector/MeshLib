#include "MRPythonViewer.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRMouseController.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRSetupViewer.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRStringConvert.h"
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#ifdef __APPLE__
#include <pthread.h>
#include <sys/select.h>
#include <unistd.h>

// From the interpreter (`pythonrun.h`), linked lazily (`-U` in `CMakeLists.txt`): this library has no other use of Python.
extern "C" int ( *PyOS_InputHook )( void );
#endif

namespace MR::PythonViewer
{

static_assert( int( KeyMod::Ctrl ) == GLFW_MOD_CONTROL && int( KeyMod::Super ) == GLFW_MOD_SUPER && int( KeyMod::Shift ) == GLFW_MOD_SHIFT && int( KeyMod::Alt ) == GLFW_MOD_ALT );
static_assert( int( ViewerLaunchParamsMode::NoWindow ) == MR::LaunchParams::NoWindow );
static_assert( int( ViewportFitMode::SelectedObjects ) == int( MR::FitMode::SelectedObjects ) );

namespace
{

// The module releases the GIL around the calls, so these can wait for the GUI thread.
template <typename F>
void runInGuiThread( F&& f )
{
    CommandLoop::runCommandFromGUIThread( std::forward<F>( f ) );
}

/// this viewer draws no ribbon, and the libraries owning the ribbon items are not loaded here,
/// so reading the schema of their items would only warn that every one of them is not registered
class MinimalRibbonMenu final : public RibbonMenu
{
    void readMenuItemsStructure_() override {}
};

class MinimalViewerSetup final : public MR::ViewerSetup
{
public:
    void setupBasePlugins( MR::Viewer* viewer ) const override
    {
        auto menu = std::make_shared<MinimalRibbonMenu>();
        menu->setMenuUIConfig( { .topLayout = RibbonTopPanelLayoutMode::None,.drawScenePanel = false,.drawToolbar = false } ); // no scene tree by default
        viewer->setMenuPlugin( menu );
    }
    void setupExtendedLibraries() const override {}
    void unloadExtendedLibraries() const override {}

    void setupConfiguration( MR::Viewer* viewer ) const override
    {
        viewer->resetSettingsFunction = [base = viewer->resetSettingsFunction] ( MR::Viewer* viewer )
        {
            base( viewer );
            resetSettings_( viewer );
        };
        viewer->resetSettingsFunction( viewer );
    }

private:
    static void resetSettings_( MR::Viewer* viewer )
    {
        viewer->glPickRadius = 3;

        auto& mouseController = viewer->mouseController();
        mouseController.setMouseControl( { MR::MouseButton::Right, 0 }, MouseMode::Translation );
        mouseController.setMouseControl( { MR::MouseButton::Middle, 0 }, MouseMode::Rotation );
        mouseController.setMouseControl( { MR::MouseButton::Middle, getGlfwModPrimaryCtrl() }, MouseMode::Roll );
    }
};

MR::Viewer::LaunchParams toLaunchParams( const ViewerLaunchParams& p )
{
    MR::Viewer::LaunchParams ret;
    ret.animationMaxFps = p.animationMaxFps;
    ret.fullscreen = p.fullscreen;
    ret.width = p.width;
    ret.height = p.height;
    ret.isAnimating = p.isAnimating;
    ret.name = p.name;
    ret.windowMode = MR::LaunchParams::WindowMode( p.windowMode );
    return ret;
}

#ifndef __APPLE__
// the GUI thread's exit code, for `showViewer()` to wait on
std::shared_future<int> gViewerFinished;
#else
// the launch params and setup, for the phases that follow launch()
std::shared_ptr<MR::Viewer::LaunchParams> gLaunchParams;
std::shared_ptr<MinimalViewerSetup> gLaunchSetup;

// AppKit runs a GUI on the process main thread and nowhere else, so here the viewer lives on the
// calling thread: launch() pre-launches it and returns, the calls run inline, and the window appears
// and is pumped by showViewer() or, while the interpreter waits for a line of terminal input, by the
// input hook below.
int ( *gPreviousInputHook )() = nullptr;
int pumpViewerInputHook();

void setupInputHook()
{
    gPreviousInputHook = PyOS_InputHook;
    PyOS_InputHook = pumpViewerInputHook;
}

void removeInputHook()
{
    if ( PyOS_InputHook == pumpViewerInputHook )
        PyOS_InputHook = gPreviousInputHook;
}

bool stdinHasData()
{
    fd_set fds;
    FD_ZERO( &fds );
    FD_SET( STDIN_FILENO, &fds );
    timeval noWait{};
    // an error counts as data: stop pumping and let the interpreter's own read fail
    return select( STDIN_FILENO + 1, &fds, nullptr, nullptr, &noWait ) != 0;
}

// Called by the interpreter, without the GIL, while it waits for a line of terminal input at the
// prompt or in input(): pumps the viewer until the line arrives, so the window stays live meanwhile.
int pumpViewerInputHook()
{
    while ( !stdinHasData() )
    {
        auto rc = protectedLaunchPhase( []
        {
            auto& viewer = getViewerInstance();
            assert( viewer.isPreLaunched() );

            if ( !viewer.isLaunched() )
                if ( auto rc = viewer.launchShow( *gLaunchParams ); rc != EXIT_SUCCESS )
                    return rc;

            // wakes on any window event, else re-checks stdin 20 times a second
            viewer.runEventLoopIteration( 0.05 );

            if ( viewer.windowShouldClose() )
            {
                viewer.launchShut();
                return EXIT_FAILURE;
            }

            return EXIT_SUCCESS;
        } );
        if ( rc != EXIT_SUCCESS )
        {
            postLaunchDefaultViewer( *gLaunchParams, *gLaunchSetup );
            removeInputHook();
            break;
        }
    }
    return gPreviousInputHook ? gPreviousInputHook() : 0;
}
#endif

} // namespace

KeyMod getPrimaryCtrlKeyMod()
{
    return KeyMod( getGlfwModPrimaryCtrl() );
}

void Viewport::cameraLookAlong( const Vector3f& dir, const Vector3f& up ) const
{
    runInGuiThread( [&] { getViewerInstance().viewport( id ).cameraLookAlong( dir, up ); } );
}

void Viewport::cameraRotateAround( const Line3f& axis, float angle ) const
{
    runInGuiThread( [&] { getViewerInstance().viewport( id ).cameraRotateAround( axis, angle ); } );
}

Vector3f Viewport::projectToViewportSpace( const Vector3f& input ) const
{
    Vector3f ret;
    runInGuiThread( [&] { ret = getViewerInstance().viewport( id ).projectToViewportSpace( input ); } );
    return ret;
}

Viewport Viewer::viewport( ViewportId viewportId ) const
{
    return Viewport( getViewerInstance().viewport( viewportId ).id );
}

void Viewer::incrementForceRedrawFrames( int num, bool swapOnLastOnly ) const
{
    runInGuiThread( [&] { getViewerInstance().incrementForceRedrawFrames( num, swapOnLastOnly ); } );
}

void Viewer::skipFrames( int frames ) const
{
    while ( frames > 0 )
    {
        frames--;
        runInGuiThread( [] {} );
    }
}

void Viewer::preciseFitDataViewport( ViewportMask vpList, const ViewportFitDataParams& params ) const
{
    runInGuiThread( [&]
    {
        getViewerInstance().preciseFitDataViewport( vpList, FitDataParams( params.factor, params.snapView, FitMode( params.mode ) ) );
    } );
}

void Viewer::captureScreenShot( const std::string& path ) const
{
    runInGuiThread( [&] ()
    {
        auto image = getViewerInstance().captureSceneScreenShot();
        (void)ImageSave::toAnySupportedFormat( image, path ); //TODO: process potential error
    } );
}

void Viewer::captureUIScreenShot( const std::string& path ) const
{
    auto filename = pathFromUtf8( path );
    runInGuiThread( [filename] ()
    {
        getViewerInstance().captureUIScreenShot( [filename] ( const Image& image )
        {
            (void)ImageSave::toAnySupportedFormat( image, filename ); //TODO: process potential error
        } );
    } );
}

void Viewer::shutdown( int exitCode, bool interruptible ) const
{
    runInGuiThread( [&] { getViewerInstance().stopEventLoop( exitCode, interruptible ); } );
}

void Viewer::mouseDown( MouseButton button, KeyMod modifier ) const
{
    auto& v = getViewerInstance();
    v.emplaceEvent( "simulatedMouseDown", [&v, button, modifier]
    {
        v.mouseDown( MR::MouseButton( button ), int( modifier ) );
    } );
}

void Viewer::mouseUp( MouseButton button, KeyMod modifier ) const
{
    auto& v = getViewerInstance();
    v.emplaceEvent( "simulatedMouseUp", [&v, button, modifier]
    {
        v.mouseUp( MR::MouseButton( button ), int( modifier ) );
    } );
}

void Viewer::mouseMove( int x, int y ) const
{
    runInGuiThread( [x, y]
    {
        auto& viewer = getViewerInstance();
        glfwSetCursorPos( viewer.window, double( x ) / viewer.pixelRatio, double( y ) / viewer.pixelRatio );

        // On Windows `glfwSetCursorPos()` automatically sends the `mouseMove()` event. On Linux it doesn't, so we need this:
        auto eventCall = [&viewer, x, y]{ viewer.mouseMove( x, y ); };
        viewer.emplaceEvent( "simulatedMouseMove", eventCall, false );
    } );
}

Vector2f Viewer::getMousePos() const
{
    double x = -1, y = -1;
    runInGuiThread( [&x, &y]
    {
        const auto& v = getViewerInstance();
        if ( v.window )
        {
            glfwGetCursorPos( v.window, &x, &y );
            x *= v.pixelRatio;
            y *= v.pixelRatio;
        }
    } );
    return Vector2f( float( x ), float( y ) );
}

Vector3f Viewer::viewportToScreen( const Vector3f& viewportPoint, ViewportId id ) const
{
    return getViewerInstance().viewportToScreen( viewportPoint, id );
}

void Viewer::showSceneTree( bool show ) const
{
    runInGuiThread( [show]
    {
        auto& viewer = getViewerInstance();
        if ( auto ribbonMenu = viewer.getMenuPluginAs<RibbonMenu>() )
        {
            auto config = RibbonMenuUIConfig();
            config.topLayout = RibbonTopPanelLayoutMode::None;
            config.drawToolbar = false;
            config.drawScenePanel = show;
            ribbonMenu->setMenuUIConfig( config );
            viewer.incrementForceRedrawFrames( viewer.forceRedrawMinimumIncrementAfterEvents, viewer.swapOnLastPostEventsRedraw );
        }
    } );
}

// The viewer on a detached thread; the caller continues and drives it with blocking calls.
void launch( const ViewerLaunchParams& pythonParams, const ViewerSetup& )
{
    const auto params = toLaunchParams( pythonParams );
    MinimalViewerSetup setup;
#ifndef __APPLE__
    std::promise<int> launchedPromise;
    std::promise<int> finishedPromise;
    auto launched = launchedPromise.get_future();
    auto finished = finishedPromise.get_future();

    std::thread guiThread { [params, setup, launched = std::move( launchedPromise ), finished = std::move( finishedPromise )] () mutable
    {
        SetCurrentThreadName( "PythonAppLaunchThread" );

        const auto exitCode = preLaunchDefaultViewer( params, setup );
        launched.set_value( exitCode );
        if ( exitCode == EXIT_SUCCESS )
            finished.set_value( launchDefaultViewer( params, setup ) );
    } };
    guiThread.detach();

    if ( auto exitCode = launched.get(); exitCode != EXIT_SUCCESS )
    {
        throw std::runtime_error(
            "Viewer could not start: glfwInit failed (no display available?), or the viewer was already "
            "launched once in this process; exit code " + std::to_string( exitCode )
        );
    }

    gViewerFinished = std::move( finished );
#else
    // more info: https://stackoverflow.com/questions/74893322
    if ( !pthread_main_np() )
        throw std::runtime_error( "This function must be called from the main thread on macOS, the only thread a GUI can run on" );

    gLaunchParams = std::make_shared<MR::Viewer::LaunchParams>( params );
    gLaunchSetup = std::make_shared<MinimalViewerSetup>( setup );
    if ( params.windowMode == MR::LaunchParams::Show )
        gLaunchParams->windowMode = MR::LaunchParams::HideInit;

    if ( auto exitCode = preLaunchDefaultViewer( *gLaunchParams, *gLaunchSetup ); exitCode != EXIT_SUCCESS )
    {
        throw std::runtime_error(
            "Viewer could not start: glfwInit failed (no display available?), or the viewer was already "
            "launched once in this process; exit code " + std::to_string( exitCode )
        );
    }

    setupInputHook();
#endif
}

// the window is up since launch(); this waits until the user closes it or shutdown() is called
void showViewer()
{
    auto& viewer = getViewerInstance();
    if ( !viewer.isPreLaunched() )
        throw std::runtime_error( "Viewer is not launched: call launch() first" );

    int exitCode;
#ifndef __APPLE__
    exitCode = gViewerFinished.get();
#else
    // more info: https://stackoverflow.com/questions/74893322
    if ( !pthread_main_np() )
        throw std::runtime_error( "This function must be called from the main thread on macOS, the only thread a GUI can run on" );

    exitCode = protectedLaunchPhase( []
    {
        auto& viewer = getViewerInstance();
        if ( !viewer.isLaunched() )
            if ( auto rc = viewer.launchShow( *gLaunchParams ); rc != EXIT_SUCCESS )
                return rc;
        viewer.launchEventLoop();
        viewer.launchShut();
        return EXIT_SUCCESS;
    } );
    postLaunchDefaultViewer( *gLaunchParams, *gLaunchSetup );

    removeInputHook();
#endif
    if ( exitCode != EXIT_SUCCESS )
        throw std::runtime_error( "Viewer failed with exit code " + std::to_string( exitCode ) );
}

void runFromGUIThread( const std::function<void()>& lambda )
{
    // a Python callable takes the GIL itself, and an exception from it is rethrown here
    runInGuiThread( [&lambda] { lambda(); } );
}

// The interpreter is exiting with the viewer still up: close it the way a closed window would, so the
// settings are saved and the plugins unload, instead of dying mid-loop.
void shutViewerAtExit()
{
#ifndef __APPLE__
    if ( !gViewerFinished.valid() || gViewerFinished.wait_for( std::chrono::seconds( 0 ) ) == std::future_status::ready )
        return;
    try
    {
        CommandLoop::runCommandFromGUIThread( [] { getViewerInstance().stopEventLoop(); } );
    }
    catch ( const std::exception& )
    {
        // the loop is already on its way out, but its teardown (e.g. MCP server shutdown)
        // must still finish before static destructors run, or both free the same objects
    }
    gViewerFinished.wait_for( std::chrono::seconds( 10 ) ); // a stuck loop must not hold the exit forever
#else
    if ( !gLaunchParams ) // the viewer was not launched by this module, e.g. Python embedded in an application
        return;
    auto& viewer = getViewerInstance();
    if ( viewer.isPreLaunched() )
        postLaunchDefaultViewer( *gLaunchParams, *gLaunchSetup );
    removeInputHook();
#endif
}

} // namespace MR::PythonViewer
