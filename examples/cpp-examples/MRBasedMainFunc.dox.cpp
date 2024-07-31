/// \page MRBasedMainFunc Basing your application on MeshLib
///
/// First of all you should make `main` function like this
///
/// \code
#include <MRMesh/MRLog.h>
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
/// \endcode
///
/// Then you should make your plugins, to find how have a look at \ref StatePluginsHelp page
