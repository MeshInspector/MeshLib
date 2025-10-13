#include <MRMesh/MRLog.h>
#include <MRMesh/MRSystem.h>
#include <MRViewer/MRSetupViewer.h>
#include <MRViewer/MRViewer.h>

int main( int argc, char** argv )
{
    MR::Viewer::LaunchParams launchParams{ .argc = argc, .argv = argv };
    MR::Viewer::parseLaunchParams( launchParams );
    launchParams.name = "Your app name";

    return MR::launchDefaultViewer( launchParams, MR::ViewerSetup() );
}
