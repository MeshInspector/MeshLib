#include "MRMesh/MRSystem.h"
#include "MRMesh/MRSystemPath.h"
#include "MRPython/MRPython.h"
#include "MREmbeddedPython/MREmbeddedPython.h"
#include "MRPch/MRSpdlog.h"
#include <cstdlib>

#ifdef _WIN32
#include "MRPch/MRWinapi.h"
#else
#include <dlfcn.h>
#endif

int main()
{
    MR::setupLoggerByDefault();

    // Load mrmeshpy. We do it here instead of linking against it for two reasons:
    // 1. To allow not building the Python modules.
    // 2. To allow building them separately, after this executable.
#ifdef _WIN32
    auto lib = LoadLibraryA( "mrmeshpy.pyd" );
    if ( !lib )
    {
        spdlog::error( "Unable to load the Python module mrmeshpy.pyd error: {}", GetLastError() );
        std::exit(1);
    }
#else // if not on Windows:
    auto mrmeshpyPath = MR::SystemPath::getExecutablePath().value().parent_path() / "meshlib/mrmeshpy.so";
    auto lib = dlopen( mrmeshpyPath.c_str(), RTLD_NOW | RTLD_GLOBAL );
    if ( !lib )
    {
        spdlog::error( "Unable to load the Python module {} error: {}", mrmeshpyPath.c_str(), dlerror() );
        std::exit(1);
    }
#endif

    //Test python mrmeshpy
    auto str = "import mrmeshpy\n"
        "print( \"List of python module functions available in mrmeshpy:\\n\" )\n"
        "funcs = dir( mrmeshpy )\n"
        "for f in funcs :\n"
        " if not f.startswith( '_' ) :\n"
        "  print( \"mrmeshpy.\" + f )\n"
        "print()"; // one empty line

    spdlog::info( "Running embedded python" );
    bool ok = MR::EmbeddedPython::runString( str );
    if ( ok )
        spdlog::info( "Embedded python run passed" );
    else
        spdlog::error( "Embedded python run failed" );
    MR::EmbeddedPython::shutdown();
    spdlog::info( "Embedded python shut down" );

    if ( !ok )
        return 1;

    if ( StderrPyRedirector::getNumWritten() > 0 )
    {
        spdlog::error( "Some errors reported from python" );
        return 1;
    }
    return 0;
}
