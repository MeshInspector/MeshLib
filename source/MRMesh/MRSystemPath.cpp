#include "MRSystemPath.h"
#include "MROnInit.h"

#ifndef __EMSCRIPTEN__
#define BOOST_DLL_USE_STD_FS
#include <boost/dll/runtime_symbol_info.hpp>
#endif

namespace
{

#if !defined( _WIN32 ) && !defined( __EMSCRIPTEN__ )
// If true, the resources should be loaded from the executable directory, rather than from the system directories.
[[nodiscard]] bool resourcesAreNearExe()
{
    auto opt = std::getenv( "MR_LOCAL_RESOURCES" );
    return opt && std::string_view( opt ) == "1";
}
#endif

std::filesystem::path defaultResourcesDirectory()
{
#if defined( __EMSCRIPTEN__ )
    return "/";
#elif defined( _WIN32 )
    return MR::SystemPath::getExecutableDirectory().value_or( "\\" );
#elif defined( __APPLE__ )
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return MR::SystemPath::getExecutableDirectory().value_or( "/" );
    return MR::SystemPath::getLibraryDirectory().value_or( "/" ) / ".." / "Resources";
#else
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return MR::SystemPath::getExecutableDirectory().value_or( "/" );
    return "/usr/local/etc/" + std::string( MR_PROJECT_NAME ) + "/";
#endif
}

std::filesystem::path defaultFontsDirectory()
{
#if defined( __EMSCRIPTEN__ )
    return "/";
#elif defined( _WIN32 )
    return MR::SystemPath::getExecutableDirectory().value_or( "\\" );
#elif defined( __APPLE__ )
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return MR::SystemPath::getExecutableDirectory().value_or( "/" );
    return MR::SystemPath::getLibraryDirectory().value_or( "/" ) / ".." / "Resources" / "fonts";
#else
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return MR::SystemPath::getExecutableDirectory().value_or( "/" );
    return "/usr/local/share/fonts/";
#endif
}

std::filesystem::path defaultPluginsDirectory()
{
#if defined( __EMSCRIPTEN__ )
    return "/";
#elif defined( _WIN32 )
    return MR::SystemPath::getExecutableDirectory().value_or( "\\" );
#elif defined( __APPLE__ )
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return MR::SystemPath::getExecutableDirectory().value_or( "/" );
    return MR::SystemPath::getLibraryDirectory().value_or( "/" );
#else
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return MR::SystemPath::getExecutableDirectory().value_or( "/" );
    return "/usr/local/lib/" + std::string( MR_PROJECT_NAME ) + "/";
#endif
}

std::filesystem::path defaultPythonModulesDirectory()
{
#if defined( __EMSCRIPTEN__ )
    return "/";
#elif defined( _WIN32 )
    return MR::SystemPath::getExecutableDirectory().value_or( "\\" );
#elif defined( __APPLE__ )
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return MR::SystemPath::getExecutableDirectory().value_or( "/" );
    return MR::SystemPath::getLibraryDirectory().value_or( "/" ) / ".." / "Frameworks";
#else
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return MR::SystemPath::getExecutableDirectory().value_or( "/" );
    return "/usr/local/lib/" + std::string( MR_PROJECT_NAME ) + "/";
#endif
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
    MR::SystemPath::overrideDirectory( MR::SystemPath::Directory::Resources, defaultResourcesDirectory() );
    MR::SystemPath::overrideDirectory( MR::SystemPath::Directory::Fonts, defaultFontsDirectory() );
    MR::SystemPath::overrideDirectory( MR::SystemPath::Directory::Plugins, defaultPluginsDirectory() );
    MR::SystemPath::overrideDirectory( MR::SystemPath::Directory::PythonModules, defaultPythonModulesDirectory() );
};
