#include "MRMesh/MRSystem.h"
#include "MRViewer/MRGetSystemInfoJson.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRSpdlog.h"

int main()
{
    MR::setupLoggerByDefault();

    // print compiler info
#ifdef __clang__
    spdlog::info( "{}", __VERSION__ );
#elif defined __GNUC__
    spdlog::info( "GCC {}", __VERSION__ );
#else
    spdlog::info( "MSVC {}", _MSC_FULL_VER );
#endif

    // print standard library info
#ifdef _MSVC_STL_UPDATE
    // https://github.com/microsoft/STL/wiki/Macro-_MSVC_STL_UPDATE
    spdlog::info( "Microsoft's STL version {}", _MSVC_STL_UPDATE );
#endif
#ifdef __GLIBCXX__
    spdlog::info( "GNU libstdc++ version {}", __GLIBCXX__ );
#endif
#ifdef _LIBCPP_VERSION
    spdlog::info( "Clang's libc++ version {}", _LIBCPP_VERSION );
#endif

    spdlog::info( "System info:\n{}", MR::GetSystemInfoJson().toStyledString() );
    MR::CommandLoop::removeCommands( false ); // that are added there by plugin constructors
    return 0;
}
