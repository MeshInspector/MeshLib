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
    pybind11::module_::import( "mrmeshpy" );
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
        value( "Hide", MR::Viewer::LaunchParams::WindowMode::Hide ).
        value( "HideInit", MR::Viewer::LaunchParams::WindowMode::HideInit ).
        value( "Show", MR::Viewer::LaunchParams::WindowMode::Show );

    pybind11::class_<MR::Viewer::LaunchParams>( m, "ViewerLaunchParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "animationMaxFps", &MR::Viewer::LaunchParams::animationMaxFps ).
        def_readwrite( "fullscreen", &MR::Viewer::LaunchParams::fullscreen ).
        def_readwrite( "width", &MR::Viewer::LaunchParams::width ).
        def_readwrite( "height", &MR::Viewer::LaunchParams::height ).
        def_readwrite( "isAnimating", &MR::Viewer::LaunchParams::isAnimating ).
        def_readwrite( "name", &MR::Viewer::LaunchParams::name ).
        def_readwrite( "windowMode", &MR::Viewer::LaunchParams::windowMode );

    pybind11::class_<MR::Viewport>( m, "Viewport" ).
        def( "cameraLookAlong", MR::pythonRunFromGUIThread( &MR::Viewport::cameraLookAlong ), "changes camera look, ensure calling fit_data after this function" ).
        def( "cameraRotateAround", MR::pythonRunFromGUIThread( &MR::Viewport::cameraRotateAround ), "changes camera look, ensure calling fit_data after this function" );
    
    pybind11::enum_<MR::Viewport::FitMode>( m, "ViewportFitMode" ).
        value( "Visible", MR::Viewport::FitMode::Visible ).
        value( "SelectedObjects", MR::Viewport::FitMode::SelectedObjects ).
        value( "SelectedPrimitives", MR::Viewport::FitMode::SelectedPrimitives );

    pybind11::class_<MR::Viewport::FitDataParams>( m, "ViewportFitDataParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "factor", &MR::Viewport::FitDataParams::factor ).
        def_readwrite( "snapView", &MR::Viewport::FitDataParams::snapView ).
        def_readwrite( "mode", &MR::Viewport::FitDataParams::mode );

    pybind11::class_<MR::Viewer, std::unique_ptr<MR::Viewer, pybind11::nodelete>>( m, "Viewer" ).
        def( pybind11::init( [] ()
    {
        return std::unique_ptr<MR::Viewer, pybind11::nodelete>( MR::Viewer::instance() );
    } ) ).
        def( "viewport", ( MR::Viewport& ( MR::Viewer::* )( MR::ViewportId ) )& MR::Viewer::viewport,
            pybind11::arg( "viewportId" ) = MR::ViewportId{}, pybind11::return_value_policy::reference_internal ).
        def( "incrementForceRedrawFrames", MR::pythonRunFromGUIThread( &MR::Viewer::incrementForceRedrawFrames ),
            pybind11::arg( "num" ) = 1,
            pybind11::arg( "swapOnLastOnly" ) = false ).
        def( "preciseFitDataViewport", MR::pythonRunFromGUIThread( &MR::Viewer::preciseFitDataViewport ),
            pybind11::arg( "vpList" ) = MR::ViewportMask::all(),
            pybind11::arg( "params" ) = MR::Viewport::FitDataParams() ).
        def( "captureScreenShot", &pythonCaptureScreenShot ).
      def( "shutdown", MR::pythonRunFromGUIThread( &MR::Viewer::stopEventLoop ) );

    m.def( "launch", &pythonLaunch, "launches viewer" );
} )