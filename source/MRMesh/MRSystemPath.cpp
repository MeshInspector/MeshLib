#include "MRSystemPath.h"
#include "MROnInit.h"

#ifndef __EMSCRIPTEN__
#define BOOST_DLL_USE_STD_FS
#include <boost/dll/runtime_symbol_info.hpp>
#endif

namespace
{

using namespace MR;

#if !defined( _WIN32 ) && !defined( __EMSCRIPTEN__ )
// If true, the resources should be loaded from the executable directory, rather than from the system directories.
[[nodiscard]] bool resourcesAreNearExe()
{
    auto opt = std::getenv( "MR_LOCAL_RESOURCES" );
    return opt && std::string_view( opt ) == "1";
}
#endif

std::filesystem::path defaultDirectory( SystemPath::Directory dir )
{
    using Directory = SystemPath::Directory;
#if defined( __EMSCRIPTEN__ )
    return "/";
#elif defined( _WIN32 )
    return SystemPath::getExecutableDirectory().value_or( "\\" );
#elif defined( __APPLE__ )
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return SystemPath::getExecutableDirectory().value_or( "/" );

    const auto libDir = SystemPath::getLibraryDirectory().value_or( "/" );
    switch ( dir )
    {
        case Directory::Resources:
            return libDir / ".." / "Resources";
        case Directory::Fonts:
            return libDir / ".." / "Resources" / "fonts";
        case Directory::Plugins:
            return libDir;
        case Directory::PythonModules:
            return libDir / ".." / "Frameworks";
        case Directory::Count:
            MR_UNREACHABLE
    }
#else
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return SystemPath::getExecutableDirectory().value_or( "/" );

    const auto libDir = SystemPath::getLibraryDirectory().value_or( "/" );
    const auto installDir = libDir / ".." / "..";
    switch ( dir )
    {
        case Directory::Resources:
            return installDir / "etc" / MR_PROJECT_NAME;
        case Directory::Fonts:
            return installDir / "share" / "fonts";
        case Directory::Plugins:
        case Directory::PythonModules:
            return installDir / "lib" / MR_PROJECT_NAME;
        case Directory::Count:
            MR_UNREACHABLE
    }
#endif
    MR_UNREACHABLE
}

} // namespace

namespace MR
{

Expected<std::filesystem::path> SystemPath::getExecutablePath()
{
#ifndef __EMSCRIPTEN__
    std::error_code ec;
    auto result = boost::dll::program_location( ec );
    if ( ec )
        return unexpected( "Failed to get executable path: " + ec.message() );
    return result;
#else
    return unexpected( "Not supported on Wasm" );
#endif
}

Expected<std::filesystem::path> SystemPath::getLibraryPath()
{
#ifndef __EMSCRIPTEN__
    std::error_code ec;
    auto result = boost::dll::this_line_location( ec );
    if ( ec )
        return unexpected( "Failed to get library path: " + ec.message() );
    return result;
#else
    return unexpected( "Not supported on Wasm" );
#endif
}

Expected<std::filesystem::path> SystemPath::getExecutableDirectory()
{
    return getExecutablePath().transform( [] ( auto&& path ) { return path.parent_path(); } );
}

Expected<std::filesystem::path> SystemPath::getLibraryDirectory()
{
    return getLibraryPath().transform( [] ( auto&& path ) { return path.parent_path(); } );
}

SystemPath& SystemPath::instance_()
{
    static SystemPath instance;
    return instance;
}

std::filesystem::path SystemPath::getDirectory( SystemPath::Directory dir )
{
    return instance_().directories_[(size_t)dir];
}

void SystemPath::overrideDirectory( SystemPath::Directory dir, const std::filesystem::path& path )
{
    instance_().directories_[(size_t)dir] = path;
}

} // namespace MR

MR_ON_INIT
{
    using namespace MR;
    for ( auto dir = 0; dir < (int)SystemPath::Directory::Count; ++dir )
        SystemPath::overrideDirectory( SystemPath::Directory( dir ), defaultDirectory( SystemPath::Directory( dir ) ) );
};
