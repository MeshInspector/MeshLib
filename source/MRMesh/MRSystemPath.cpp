#include "MRSystemPath.h"
#include "MROnInit.h"
#include "MRDirectory.h"

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

const std::vector<SystemPath::SystemFontPaths>& SystemPath::getSystemFonts()
{
    static std::vector<SystemPath::SystemFontPaths> fonts;
    if ( !fonts.empty() )
    {
        return fonts;
    }

    std::filesystem::path systemFontspath;
#ifdef _WIN32
    //systemFontspath = "C:/Windows/Fonts";
    systemFontspath = "C:/all/font";

#elif defined (__APPLE__)
    path = "Library/Fonts";
#else // linux and wasm
    path = "/usr/share/fonts";
    static const std::vector<std::string> suffixes{
            "bold",
            "Bold",
            "light",
            "Light",
            "medium",
            "Medium",
            "regular",
            "Regular",
            "SemiBold",
            "BoldItalic",
            "italic",
            "Italic",
            "LightItalic",
            "MediumItalic",
            "SemiBold",
            "SemiBoldItalic",
            "BoldOblique",
            "oblique",
            "Oblique"
    };
#endif
    std::string firstFontName;
    std::string curName;
    std::error_code ec;

    for ( auto entry : MR::Directory{ systemFontspath, ec } )
    {
        std::filesystem::path font = entry;
        if ( font.extension() != ".ttf" )
        {
            continue;
        }
        
        curName = font.stem().string();

#ifdef _WIN32
        if ( firstFontName.empty() || curName.find( firstFontName ) == std::string::npos )
        {
            firstFontName = curName;
            fonts.push_back( SystemFontPaths() );
            fonts.back()[( size_t )SystemFontType::Regular] = font;
            continue;
        }
        if ( curName == firstFontName + "b" || curName == firstFontName + "B" )
        {
            fonts.back()[( size_t )SystemFontType::Bold] = font;
        }
        if ( curName == firstFontName + "i" || curName == firstFontName + "I" )
        {
            fonts.back()[( size_t )SystemFontType::Italic] = font;
        }
        if ( curName == firstFontName + "bi" || curName == firstFontName + "BI" )
        {
            fonts.back()[( size_t )SystemFontType::BoldItalic] = font;
        }
#elif defined (__APPLE__)
        // TO DO
#else // linux and wasm
        if ( firstFontName.empty() || curName.find( firstFontName ) == std::string::npos )
        {
            bool findSuffix = false;
            for ( const auto& suffix : suffixes )
            {
                auto pos = curName.find( "-" + suffix );
                if ( pos != std::string::npos )
                {
                    firstFontName = std::string( curName.begin(), curName.begin() + pos );
                    findSuffix = true;
                    break;
                }
                pos = curName.find( suffix );
                if ( pos != std::string::npos )
                {
                    firstFontName = std::string( curName.begin(), curName.begin() + pos );
                    findSuffix = true;
                    break;
                }
            }
            if ( !findSuffix )
            {
                firstFontName = curName;
            }

            fonts.push_back( SystemFontPaths() );
        }

        if ( curName.find( "regular" ) != std::string::npos || curName.find( "-regular" ) != std::string::npos ||
             curName.find( "Regular" ) != std::string::npos || curName.find( "-Regular" ) != std::string::npos )
        {
            fonts.back()[( size_t )SystemFontType::Regular] = font;
        }
        else if ( curName.find( "BoldItalic" ) != std::string::npos || curName.find( "-BoldItalic" ) != std::string::npos )
        {
            fonts.back()[( size_t )SystemFontType::BoldItalic] = font;
        }
        else if ( curName.find( "bold" ) != std::string::npos || curName.find( "-bold" ) != std::string::npos ||
             curName.find( "Bold" ) != std::string::npos || curName.find( "-Bold" ) != std::string::npos )
        {
            fonts.back()[( size_t )SystemFontType::Bold] = font;
        }
        else if ( curName.find( "italic" ) != std::string::npos || curName.find( "-italic" ) != std::string::npos ||
             curName.find( "Italic" ) != std::string::npos || curName.find( "-Italic" ) != std::string::npos )
        {
            fonts.back()[( size_t )SystemFontType::Italic] = font;
        }
        else if ( curName == firstFontName )
        {
            fonts.back()[( size_t )SystemFontType::Regular] = font;
        }
#endif
        
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
