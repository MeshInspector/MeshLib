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
#if defined( __EMSCRIPTEN__ )
    (void)dir;
    return "/";
#elif defined( _WIN32 )
    (void)dir;
    return SystemPath::getExecutableDirectory().value_or( "\\" );
#elif defined( __APPLE__ )
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return SystemPath::getExecutableDirectory().value_or( "/" );

    const auto libDir = SystemPath::getLibraryDirectory().value_or( "/" );
    using Directory = SystemPath::Directory;
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
    MR_UNREACHABLE
#else
    // TODO: use getLibraryDirectory()
    if ( resourcesAreNearExe() )
        return SystemPath::getExecutableDirectory().value_or( "/" );

    const std::filesystem::path installDir ( "/usr/local/" );
    using Directory = SystemPath::Directory;
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
    MR_UNREACHABLE
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

std::filesystem::path SystemPath::getSystemFontsDirectory()
{
    std::filesystem::path path;
#if defined(__EMSCRIPTEN__)
    path = "/usr/share/fonts";
#elif defined(__APPLE__)
    path = "Library/Fonts"
#else
    path = "C:/Windows/Fonts";
#endif
    return path;
}

const std::vector<std::string>& SystemPath::getAllSystemFonts()
{
    static std::vector<std::string> allFonts;
    if ( !allFonts.empty() )
    {
        return allFonts;
    }

    auto path = getSystemFontsDirectory();
    for ( auto& p : std::filesystem::directory_iterator( path ) )
    {
        allFonts.push_back( p.path().stem().string() );
    }

    return allFonts;
}

const std::vector<std::string>& SystemPath::getSystemFonts()
{
    static std::vector<std::string> fonts;
    if ( !fonts.empty() )
    {
        return fonts;
    }
    std::vector<std::string> allFonts = getAllSystemFonts();

    std::string name;
    size_t n = 0;
    for ( const auto& font : allFonts )
    {
        if ( name.empty() )
        {
            name = font;
            continue;
        }
        if ( font.find( name ) == std::string::npos )
        {
            name = font;
            continue;
        }
        if ( font == name + "b" || font == name + "B" ||
             font == name + "i" || font == name + "I" ||
             font == name + "bi" || font == name + "BI" )
        {
            n++;
        }

        if ( n == 3 )
        {
            fonts.push_back( name );
            name.clear();
            n = 0;
        }
    }

    return fonts;
}

} // namespace MR

MR_ON_INIT
{
    using namespace MR;
    for ( auto dir = 0; dir < (int)SystemPath::Directory::Count; ++dir )
        SystemPath::overrideDirectory( SystemPath::Directory( dir ), defaultDirectory( SystemPath::Directory( dir ) ) );
};
