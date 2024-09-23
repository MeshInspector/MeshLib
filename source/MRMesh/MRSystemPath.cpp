#include "MRSystemPath.h"
#include "MROnInit.h"

#if defined( _WIN32 )
#include <libloaderapi.h>
#elif !defined( __EMSCRIPTEN__ )
#include <dlfcn.h>
#include <unistd.h>
#if defined( __APPLE__ )
#include <mach-o/dyld.h>
#else
#include <linux/limits.h>
#endif
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

#if defined( __APPLE__ )
std::filesystem::path macosPackageLibPath()
{
#ifdef MR_FRAMEWORK
    return "/Library/Frameworks/" + std::string( MR_PROJECT_NAME ) + ".framework/Versions/Current/lib";
#else
    return "/Applications/" + std::string( MR_PROJECT_NAME ) + ".app/Contents/libs";
#endif
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
    return macosPackageLibPath() / ".." / "Resources";
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
    return macosPackageLibPath() / ".." / "Resources" / "fonts";
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
    return macosPackageLibPath();
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
    return macosPackageLibPath() / ".." / "Frameworks";
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
#if defined( __EMSCRIPTEN__ )
    return unexpected( "Not supported on Wasm" );
#elif defined( _WIN32 )
    wchar_t path[MAX_PATH];
    if ( auto size = GetModuleFileNameW( NULL, path, MAX_PATH ); size == 0 )
        return unexpected( "Failed to get executable path" );
    else if ( size == MAX_PATH )
        return unexpected( "Executable path is too long" );
    return std::filesystem::path { path };
#elif defined( __APPLE__ )
    char path[PATH_MAX];
    uint32_t size = PATH_MAX + 1;
    if ( _NSGetExecutablePath( path, &size ) != 0 )
        return unexpected( "Executable path is too long" );
    path[size] = '\0';
    return std::filesystem::path { path };
#else
    char path[PATH_MAX];
    if ( auto size = readlink( "/proc/self/exe", path, PATH_MAX ); size < 0 )
        return unexpected( "Failed to get executable path" );
    else if ( size >= PATH_MAX )
        return unexpected( "Executable path is too long" );
    else
        path[size] = '\0';
    return std::filesystem::path { path };
#endif
}

Expected<std::filesystem::path> SystemPath::getLibraryPath()
{
#if defined( __EMSCRIPTEN__ )
    return unexpected( "Not supported on Wasm" );
#elif defined( _WIN32 )
    HMODULE module = NULL;
    auto rc = GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCTSTR)MR::SystemPath::getLibraryPath,
        &module
    );
    if ( !rc )
        return unexpected( "Failed to get library path" );

    wchar_t path[MAX_PATH];
    if ( auto size = GetModuleFileNameW( module, path, MAX_PATH ); size == 0 )
        return unexpected( "Failed to get library path" );
    else if ( size == MAX_PATH )
        return unexpected( "Library path is too long" );

    return std::filesystem::path { path };
#else
    Dl_info info;
    if ( !dladdr( (void*)MR::SystemPath::getLibraryPath, &info ) )
        return unexpected( "Failed to get library path" );
    return std::filesystem::path { info.dli_fname };
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
