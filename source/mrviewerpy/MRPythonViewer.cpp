#include "MRMesh/MRPython.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRViewportId.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRSetupViewer.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRSystem.h"
#include "MRViewer/MRPythonAppendCommand.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MRImageSave.h"
#include <pybind11/stl.h>
#include <memory>

MR_INIT_PYTHON_MODULE_PRECALL( mrviewerpy, [] ()
{
    try
    {
        pybind11::module_::import( "meshlib.mrmeshpy" );
    }
    catch ( const pybind11::error_already_set& pythonErr )
    {
        spdlog::warn( pythonErr.what() );
        pybind11::module_::import( "mrmeshpy" );
    }
} )

void pythonCaptureScreenShot( MR::Viewer* viewer, const char* path )
{
    MR::CommandLoop::runCommandFromGUIThread( [&] ()
    {
        auto image = viewer->captureScreenShot();
        MR::ImageSave::toAnySupportedFormat( image, path );
    } );
}

void pythonLaunch( const MR::Viewer::LaunchParams& params, const MR::ViewerSetup& setup )
{
    std::thread lauchThread = std::thread( [=] ()
    {
        MR::SetCurrentThreadName( "PythonAppLaunchThread" );
        MR::launchDefaultViewer( params, setup );
    } );
    lauchThread.detach();
}

MR_ADD_PYTHON_CUSTOM_DEF( mrviewerpy, Viewer, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::ViewerSetup>( m, "ViewerSetup" ).
        def( pybind11::init<>() );

    pybind11::enum_<MR::Viewer::LaunchParams::WindowMode>( m, "ViewerLaunchParamsMode" ).
        value( "Hide", MR::Viewer::LaunchParams::WindowMode::Hide, "Don't show window" ).
        value( "HideInit", MR::Viewer::LaunchParams::WindowMode::HideInit, "Show window after init" ).
        value( "Show", MR::Viewer::LaunchParams::WindowMode::Show, "Show window immediately" ).
        value( "TryHidden", MR::Viewer::LaunchParams::WindowMode::TryHidden, "Launches in \"Hide\" mode if OpenGL is present and \"NoWindow\" if it is not" ).
        value( "NoWindow", MR::Viewer::LaunchParams::WindowMode::NoWindow, "Don't initialize GL window (don't call GL functions)(force `isAnimating`)" );

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
            "note: this can make camera clip objects (as far as distance to scene center is not fixed)" );
    
    pybind11::enum_<MR::Viewport::FitMode>( m, "ViewportFitMode", "Fit mode ( types of objects for which the fit is applied )" ).
        value( "Visible", MR::Viewport::FitMode::Visible, "fit all visible objects" ).
        value( "SelectedObjects", MR::Viewport::FitMode::SelectedObjects, "fit only selected objects" ).
        value( "SelectedPrimitives", MR::Viewport::FitMode::SelectedPrimitives, "fit only selected primitives" );

    pybind11::class_<MR::Viewport::FitDataParams>( m, "ViewportFitDataParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "factor", &MR::Viewport::FitDataParams::factor, "part of the screen for scene location" ).
        def_readwrite( "snapView", &MR::Viewport::FitDataParams::snapView, "snapView - to snap camera angle to closest canonical quaternion" ).
        def_readwrite( "mode", &MR::Viewport::FitDataParams::mode, "fit mode" );

    pybind11::class_<MR::Viewer, std::unique_ptr<MR::Viewer, pybind11::nodelete>>( m, "Viewer", "GLFW-based mesh viewer" ).
        def( pybind11::init( [] ()
    {
        return std::unique_ptr<MR::Viewer, pybind11::nodelete>( MR::Viewer::instance() );
    } ) ).
        def( "viewport", ( MR::Viewport& ( MR::Viewer::* )( MR::ViewportId ) )& MR::Viewer::viewport,
            pybind11::arg( "viewportId" ) = MR::ViewportId{}, pybind11::return_value_policy::reference_internal,
            "Return the current viewport, or the viewport corresponding to a given unique identifier\n"
            "\tviewportId - unique identifier corresponding to the desired viewport (current viewport if 0)" ).
        def( "incrementForceRedrawFrames", MR::pythonRunFromGUIThread( &MR::Viewer::incrementForceRedrawFrames ),
            pybind11::arg( "num" ) = 1,
            pybind11::arg( "swapOnLastOnly" ) = false,
            "Increment number of forced frames to redraw in event loop\n"
            "if `swapOnLastOnly` only last forced frame will be present on screen and all previous will not" ).
        def( "preciseFitDataViewport", MR::pythonRunFromGUIThread( &MR::Viewer::preciseFitDataViewport ),
            pybind11::arg( "vpList" ) = MR::ViewportMask::all(),
            pybind11::arg( "params" ) = MR::Viewport::FitDataParams(),
            "Calls fitData and change FOV to match the screen size then\n"
            "params - params fit data" ).
        def( "captureScreenShot", &pythonCaptureScreenShot,pybind11::arg("path"),
            "Captures part of window (redraw 3d scene over UI (without redrawing UI))" ).
        def( "shutdown", MR::pythonRunFromGUIThread( &MR::Viewer::stopEventLoop ), "sets stop event loop flag (this flag is glfwShouldWindowClose equivalent)" );

    m.def( "launch", &pythonLaunch, 
        pybind11::arg( "params" ) = MR::Viewer::LaunchParams{},
        pybind11::arg( "setup" ) = MR::ViewerSetup{},
        "starts default viewer with given params and setup" );
} )