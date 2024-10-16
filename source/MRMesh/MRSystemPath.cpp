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

    std::vector<std::pair<std::filesystem::path, std::string>> allSystemFonts;
    std::vector<std::filesystem::path> systemFontspath;
#ifdef _WIN32
    //systemFontspath = "C:/Windows/Fonts";
    systemFontspath = { "C:/all/font" };

#elif defined (__APPLE__)
    systemFontspath = { "/Library/Fonts", "/System/Library/Fonts" };
#else // linux and wasm
    systemFontspath = { "/usr/share/fonts" };
#endif

    static const std::vector<std::string> suffixes{
            "regular",
            "Regular",
            "bd",
            "BD",
            "SemiBold",
            "bold",
            "b",
            "Bold",
            "B",
            "BoldItalic",
            "bi",
            "BI",
            "z",
            "LightItalic",
            "MediumItalic",
            "SemiBoldItalic",
            "li",
            "LI"
            "italic",
            "i",
            "Italic",
            "I",
            "light",
            "Light",
            "l",
            "L",
            "medium",
            "Medium",
            "BoldOblique",
            "oblique",
            "Oblique"
    };

    std::error_code ec;
    for ( auto& curPath : systemFontspath )
    {
        for ( auto entry : MR::DirectoryRecursive{ curPath, ec } )
        {
            std::filesystem::path font = entry;
            if ( font.extension() != ".ttf" )
            {
                continue;
            }

            allSystemFonts.push_back( { font, font.filename().string() } );
        }
    }

    std::sort( allSystemFonts.begin(), allSystemFonts.end(), 
        [] ( std::pair<std::filesystem::path, std::string>& v1, std::pair<std::filesystem::path, std::string>& v2 )
    {
        return v1 < v2;
    } );

    for ( auto& [font, name] : allSystemFonts )
    {
        auto pos = name.find( "arial" );
        std::string newName;
        if ( pos != std::string::npos )
        {
            name = "arial-" + std::string(name.begin() + 5, name.end() );
        }
        pos = name.find( "calibri" );
        if ( pos != std::string::npos )
        {
            name = "calibri-" + std::string( name.begin() + 7, name.end() );
        }
    }

    std::string firstFontName;
    bool newFont = false;
    size_t numFont = 0;
    for ( auto& [font, curName]: allSystemFonts )
    {
        if ( firstFontName.empty() || curName.find( firstFontName ) == std::string::npos )
        {
            fonts.push_back( SystemFontPaths() );
            newFont = true;
        }

        int numSuffix = -1;
        int n = 0;
        for ( const auto& suffix : suffixes )
        {
            auto curFontName = font.stem().string();
            auto posEndName = curName.find( "-" + suffix + ".ttf" );
            if ( posEndName != std::string::npos && curFontName != firstFontName )
            {
                if ( newFont )
                {
                    firstFontName = std::string( curName.begin(), curName.begin() + posEndName );
                    newFont = false;
                }
                numSuffix = n;
                break;
            }
            posEndName = curName.find( suffix + ".ttf" );
            if ( posEndName != std::string::npos && curFontName != firstFontName )
            {
                if ( newFont )
                {
                    firstFontName = std::string( curName.begin(), curName.begin() + posEndName );
                    newFont = false;
                }
                numSuffix = n;
                break;
            }
            n++;
        }

        if ( numSuffix < 0)
        {
            firstFontName = font.stem().string();
        }

        numFont = fonts.size() - 1;
        if ( numSuffix < 2 )
        {
            fonts[numFont][( size_t )SystemFontType::Regular] = font;
        }
        else if ( 5 <= numSuffix && numSuffix < 9 )
        {
            fonts[numFont][( size_t )SystemFontType::Bold] = font;
        }
        else if ( 18 <= numSuffix && numSuffix < 22 )
        {
            fonts[numFont][( size_t )SystemFontType::Italic] = font;
        }
        else if ( 9 <= numSuffix && numSuffix < 13 )
        {
            fonts[numFont][( size_t )SystemFontType::BoldItalic] = font;
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
