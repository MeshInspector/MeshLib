#include <MRViewer/MRSetupViewer.h>
#include <MRViewer/MRViewer.h>

int main( int argc, char** argv )
{
    MR::Viewer::LaunchParams launchParams {
        .name = "example_plugin",
        .argc = argc,
        .argv = argv,
    };
    MR::Viewer::parseLaunchParams( launchParams );

    return MR::launchDefaultViewer( launchParams, MR::ViewerSetup() );
}
