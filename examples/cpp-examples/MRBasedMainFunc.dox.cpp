#include <MRMesh/MRLog.h>
#include <MRMesh/MRSystem.h>
#include <MRViewer/MRSetupViewer.h>
#include <MRViewer/MRViewer.h>

int main( int argc, char** argv )
{
    MR::setupLoggerByDefault();

    // Init the viewer

    MR::Viewer::LaunchParams launchParams{ .argc = argc, .argv = argv };
    MR::Viewer::parseLaunchParams( launchParams );
    launchParams.name = "Your app name";

    MR::ViewerSetup viewerSetup;
    viewerSetup.setupBasePlugins( MR::Viewer::instance() );
    viewerSetup.setupCommonModifiers( MR::Viewer::instance() );
    viewerSetup.setupCommonPlugins( MR::Viewer::instance() );
    viewerSetup.setupSettingsManager( MR::Viewer::instance(), launchParams.name );
    viewerSetup.setupConfiguration( MR::Viewer::instance() );

    return MR::launchDefaultViewer( launchParams, viewerSetup );
}
