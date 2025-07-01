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
#include <pybind11/stl.h>
#include <memory>

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
    MR::CommandLoop::runCommandFromGUIThread( [func]
    {
        func();
    } );
}

namespace
{

using namespace MR;

enum class PythonKeyMod
{
    Empty = 0,
    Ctrl = GLFW_MOD_CONTROL,
    Shift = GLFW_MOD_SHIFT,
    Alt = GLFW_MOD_ALT,
};
MR_MAKE_FLAG_OPERATORS( PythonKeyMod )

/// viewer setup class for minimal configuration
/// only loads config file (if available) and configures the scene and mouse controls
class MinimalViewerSetup final : public ViewerSetup
{
public:
    void setupBasePlugins( Viewer* viewer ) const override
    {
        auto menu = std::make_shared<RibbonMenu>();
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
        mouseController.setMouseControl( { MouseButton::Middle, GLFW_MOD_CONTROL }, MouseMode::Roll );
    }
};

void pythonLaunch( const MR::Viewer::LaunchParams& params, const MinimalViewerSetup& setup )
{
    std::thread launchThread { [=]
    {
        MR::SetCurrentThreadName( "PythonAppLaunchThread" );
        MR::launchDefaultViewer( params, setup );
    } };
    launchThread.detach();
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
        "starts default viewer with given params and setup" );

    m.def( "runFromGUIThread", &pythonRunLambdaFromGUIThread, pybind11::arg( "lambda" ), "Executes given function from GUI thread, and returns after it is done" );
} )
