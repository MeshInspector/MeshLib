#include <MRMesh/MRLog.h>
#include <MRMesh/MRStringConvert.h>
#include <MRMesh/MRSystem.h>
#include <MRMesh/MRStacktrace.h>
#include <MRViewer/MRViewer.h>
#include <MRViewer/MRSetupViewer.h>
#include <MRViewer/MRSplashWindow.h>

#ifdef _WIN32
#include "MRViewer/MRConsoleWindows.h"

extern "C" __declspec( dllexport ) DWORD NvOptimusEnablement = 0x00000001;
extern "C" __declspec( dllexport ) DWORD AmdPowerXpressRequestHighPerformance = 0x00000001;

extern "C" int WINAPI WinMain(HINSTANCE /*hInstance*/, HINSTANCE /*hPrevInstance*/, char* /*lpCmdLine*/, int /*nCmdShow*/)
{
    // Initialize getting stacktraces before loading DLLs
    // https://stackoverflow.com/q/78468776/7325599
    (void)MR::getCurrentStacktraceInline();

    auto args = MR::ConvertArgv();
    std::vector<char*> argv;
    argv.reserve( args.size() );
    for ( auto& arg : args )
        argv.push_back( arg.data() );

    // Init the viewer
    MR::Viewer::LaunchParams launchParams;
    launchParams.enableTransparentBackground = true;// default false, set true for MR and MRE
    launchParams.name = std::string( MR_PROJECT_NAME );
    launchParams.splashWindow = std::make_shared<MR::DefaultSplashWindow>();
    launchParams.showMRVersionInTitle = true;
    launchParams.argc = int( argv.size() );
    launchParams.argv = argv.data();

    MR::Viewer::parseLaunchParams( launchParams );

    // Starts console if needed, free it on destructor
    MR::ConsoleRunner console( launchParams.console );

    return MR::launchDefaultViewer( launchParams, MR::ViewerSetup() );
}

#else //Unix

int main( int argc, char** argv )
{
    MR::setNewHandlerIfNeeded();

    // Init the viewer
    MR::Viewer::LaunchParams launchParams;
    launchParams.enableTransparentBackground = true;// default false, set true for MR and MRE
    launchParams.name = std::string( MR_PROJECT_NAME );
    #if !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
    launchParams.splashWindow = std::make_shared<MR::DefaultSplashWindow>();
    #endif
    launchParams.argc = argc;
    launchParams.argv = argv;

    MR::Viewer::parseLaunchParams( launchParams );

    #if defined(__APPLE__)
    setenv("XDG_DATA_DIRS", "/Library/Frameworks/MeshLib.framework/Versions/Current/share", 1);
    #endif

    return MR::launchDefaultViewer( launchParams, MR::ViewerSetup() );
}

#endif
