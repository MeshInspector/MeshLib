#include "MRSystemPath.h"
#include "MROnInit.h"
#include "MRDirectory.h"
#include "MRStringConvert.h"
#include "MRPch/MRWasm.h"
#include "MRPch/MRSpdlog.h"
#include <algorithm>
#include <map>

#if defined( _WIN32 )
#include "MRPch/MRWinapi.h"
#elif !defined( __EMSCRIPTEN__ )
#include <dlfcn.h>
#include <unistd.h>
#if defined( __APPLE__ )
#include <mach-o/dyld.h>
#else
#include <linux/limits.h>
#endif
#endif

namespace MR
{

namespace
{

std::filesystem::path defaultDirectory( SystemPath::Directory dir )
{
#if defined( __EMSCRIPTEN__ )
    (void)dir;
    return "/";
#elif defined( _WIN32 )
    (void)dir;
    return SystemPath::getExecutableDirectory().value_or( "\\" );
#elif defined( __APPLE__ )
    auto libDir = SystemPath::getLibraryDirectory().value_or( "/" );
    // detecting a developer build (all files are located in the same directory)
    const auto execDir = SystemPath::getExecutableDirectory().value_or( "/" );
    if ( libDir == execDir )
        return libDir;
    // detecting developer build's bundle (<AppName>.app/Contents/MacOS/<AppName>)
    if ( execDir.filename() == "MacOS"
        && execDir.parent_path().filename() == "Contents"
        && execDir.parent_path().parent_path().parent_path() == libDir )
        return libDir;

    using Directory = SystemPath::Directory;
    switch ( dir )
    {
        case Directory::Resources:
            return libDir.parent_path() / "Resources";
        case Directory::Fonts:
            return libDir.parent_path() / "Resources" / "fonts";
        case Directory::Plugins:
            return libDir;
        case Directory::PythonModules:
            return libDir.parent_path() / "Frameworks";
        case Directory::Count:
            MR_UNREACHABLE
    }
    MR_UNREACHABLE
#else
    auto libDir = SystemPath::getLibraryDirectory().value_or( "/" );
    // detecting a developer build (all files are located in the same directory)
    const auto execDir = SystemPath::getExecutableDirectory().value_or( "/" );
    if ( libDir == execDir )
        return libDir;

    static const auto findResourceDir = [] ( std::filesystem::path libDir ) -> std::filesystem::path
    {
        std::error_code ec;
        while ( !libDir.empty() )
        {
            const auto resourceDir = libDir / "share" / MR_PROJECT_NAME;
            if ( std::filesystem::is_directory( resourceDir, ec ) )
                return resourceDir;

            if ( libDir == "/" )
                break;
            libDir = libDir.parent_path();
        }
        // assuming this is a developer build
        return SystemPath::getExecutableDirectory().value_or( "/" );
    };

    using Directory = SystemPath::Directory;
    switch ( dir )
    {
        case Directory::Resources:
            return findResourceDir( libDir );
        case Directory::Fonts:
            return findResourceDir( libDir ) / "fonts";
        case Directory::Plugins:
        case Directory::PythonModules:
            return libDir;
        case Directory::Count:
            MR_UNREACHABLE
    }
    MR_UNREACHABLE
#endif
}

Expected<std::filesystem::path> getExecutablePath_()
{
#if defined( __EMSCRIPTEN__ )
    auto *jsStr = (char *)EM_ASM_PTR({
        var wasmStr = findWasmBinary();
        if ( wasmStr == null )
            wasmStr = "";
        var lengthBytes = lengthBytesUTF8( wasmStr ) + 1;
        var stringOnWasmHeap = _malloc( lengthBytes );
        stringToUTF8( wasmStr, stringOnWasmHeap, lengthBytes );
        return stringOnWasmHeap;
    });
    std::string wasmStr;
    if ( jsStr )
    {
        wasmStr = std::string( jsStr );
        free(jsStr);
    }
    return std::filesystem::path { wasmStr };
#elif defined( _WIN32 )
    wchar_t path[MAX_PATH];
    if ( auto size = GetModuleFileNameW( NULL, path, MAX_PATH ); size == 0 )
        return unexpected( "Failed to get executable path" );
    else if ( size == MAX_PATH )
        return unexpected( "Executable path is too long" );
    return std::filesystem::path { path };
#elif defined( __APPLE__ )
    char path[PATH_MAX];
    uint32_t size = PATH_MAX;
    if ( _NSGetExecutablePath( path, &size ) != 0 )
        return unexpected( "Executable path is too long" );
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

auto& directories_()
{
    static auto res = []
    {
        std::array<std::filesystem::path, (size_t)SystemPath::Directory::Count> dirs;
        for ( auto dir = 0; dir < (int)SystemPath::Directory::Count; ++dir )
            dirs[dir] = defaultDirectory( SystemPath::Directory( dir ) );
        return dirs;
    }();
    return res;
}

} // anonymous namespace

const Expected<std::filesystem::path>& SystemPath::getExecutablePath()
{
    static const Expected<std::filesystem::path> res = []
    {
        auto maybeRes = getExecutablePath_();
        if ( maybeRes )
        {
            spdlog::info( "Executable path: {}", utf8string( *maybeRes ) );
#ifndef __EMSCRIPTEN__ // in Wasm the path is not a file name, but starts with https://
            std::error_code ec;
            auto canonicalPath = canonical( *maybeRes, ec );
            if ( ec )
                spdlog::error( "Cannot make canonical executable path: {}", ec.message() );
            else if ( *maybeRes != canonicalPath )
            {
                *maybeRes = canonicalPath;
                spdlog::info( "Executable path in canonical form: {}", utf8string( *maybeRes ) );
            }
#endif
        }
        return maybeRes;
    }();
    return res;
}

const Expected<std::filesystem::path>& SystemPath::getLibraryPath()
{
    static const Expected<std::filesystem::path> res = []
    {
        auto maybeRes = getLibraryPathForSymbol( (void*)MR::SystemPath::getLibraryPath );
        if ( maybeRes )
        {
            spdlog::info( "Library path: {}", utf8string( *maybeRes ) );
            std::error_code ec;
            auto canonicalPath = canonical( *maybeRes, ec );
            if ( ec )
                spdlog::error( "Cannot make canonical library path: {}", ec.message() );
            else if ( *maybeRes != canonicalPath )
            {
                *maybeRes = canonicalPath;
                spdlog::info( "Library path in canonical form: {}", utf8string( *maybeRes ) );
            }
        }
        return maybeRes;
    }();
    return res;
}

Expected<std::filesystem::path> SystemPath::getLibraryPathForSymbol( const void* symbol )
{
#if defined( __EMSCRIPTEN__ )
    (void)symbol;
    return unexpected( "Not supported on Wasm" );
#elif defined( _WIN32 )
    HMODULE module = NULL;
    auto rc = GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCTSTR)symbol,
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
    if ( !dladdr( symbol, &info ) )
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

std::filesystem::path SystemPath::getDirectory( SystemPath::Directory dir )
{
    return directories_()[(size_t)dir];
}

void SystemPath::overrideDirectory( SystemPath::Directory dir, const std::filesystem::path& path )
{
    directories_()[(size_t)dir] = path;
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
    systemFontspath = { "C:/Windows/Fonts" };
#elif defined (__APPLE__)
    systemFontspath = { "/Library/Fonts", "/System/Library/Fonts" };
#else // linux and wasm
    systemFontspath = { "/usr/share/fonts" };
#endif

    static const std::map<std::string, std::vector<std::string>> typeSuffixes{
        {"regular",{"regular", "Regular"}},
        {"semibold", {"SemiBold", "semibold"}},
        {"bold", {"bold", "b", "Bold", "B", "bd", "BD"}},
        {"bolditalic", {"BoldItalic", "bolditalic", "bi", "BI", "z"}},
        {"lightitalic", {"LightItalic"}},
        {"mediumitalic", {"MediumItalic"}},
        {"semibolditalic", {"SemiBoldItalic", "semibolditalic"}},
        {"lightitalic", {"li", "LI", "LightItalic", "lightitalic"}},
        {"italic", {"italic", "i", "Italic", "I"}},
        {"light", {"light", "Light", "l", "L"}},
        {"medium", {"medium", "Medium"}},
        {"boldoblique", {"BoldOblique", "boldoblique"}},
        {"oblique", {"oblique", "Oblique"}}
    };

    std::vector<std::string> supportFormat{".ttf", ".otf"};

    std::error_code ec;
    for ( auto& curPath : systemFontspath )
    {
        for ( auto entry : MR::DirectoryRecursive{ curPath, ec } )
        {
            bool isFont = false;
            std::filesystem::path font = entry;
            for ( auto& format : supportFormat )
            {
                if ( font.extension() == format )
                {
                    isFont = true;
                    break;
                }
            }

            if( isFont )
                allSystemFonts.push_back( { font, utf8string( font.filename() ) } );
        }
    }

    std::sort( allSystemFonts.begin(), allSystemFonts.end(),
        [] ( std::pair<std::filesystem::path, std::string>& v1, std::pair<std::filesystem::path, std::string>& v2 )
    {
        return MR::toLower( v1.second ) < MR::toLower( v2.second );
    } );

    //explicit search for fonts that conflict with style suffixes
    for ( auto& [font, name] : allSystemFonts )
    {
        auto pos = name.find( "arial" );
        std::string newName;
        if ( pos != std::string::npos )
        {
            if( name[5] == '.' )
                name = "arial-regular" + std::string( name.begin() + 5, name.end() );
            else
                name = "arial-" + std::string( name.begin() + 5, name.end() );
        }
        pos = name.find( "calibri" );
        if ( pos != std::string::npos )
        {
            if ( name[7] == '.' )
                name = "calibri-regular" + std::string( name.begin() + 7, name.end() );
            else
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

        std::string suffixName;
        for ( const auto& [curSuffixName, suffixes] : typeSuffixes )
        {
            for ( const auto& suffix : suffixes )
            {
                auto curFontName = utf8string( font.stem() );
                for ( auto& format : supportFormat )
                {
                    auto posEndName = curName.find( "-" + suffix + format );
                    if ( posEndName != std::string::npos && curFontName != firstFontName )
                    {
                        if ( newFont )
                        {
                            firstFontName = std::string( curName.begin(), curName.begin() + posEndName );
                            newFont = false;
                        }
                        suffixName = curSuffixName;
                        break;
                    }
                    posEndName = curName.find( "_" + suffix + format );
                    if ( posEndName != std::string::npos && curFontName != firstFontName )
                    {
                        if ( newFont )
                        {
                            firstFontName = std::string( curName.begin(), curName.begin() + posEndName );
                            newFont = false;
                        }
                        suffixName = curSuffixName;
                        break;
                    }
                    posEndName = curName.find( suffix + format );
                    if ( posEndName != std::string::npos && curFontName != firstFontName )
                    {
                        if ( newFont )
                        {
                            firstFontName = std::string( curName.begin(), curName.begin() + posEndName );
                            newFont = false;
                        }
                        suffixName = curSuffixName;
                        break;
                    }
                }
            }
            if ( !suffixName.empty() )
            {
                break;
            }
        }

        if ( suffixName.empty() )
        {
            firstFontName = utf8string( font.stem() );
        }

        numFont = fonts.size() - 1;
        if ( suffixName == "regular" || suffixName.empty() )
        {
            fonts[numFont][( size_t )SystemFontType::Regular] = font;
        }
        else if ( suffixName == "bold" )
        {
            fonts[numFont][( size_t )SystemFontType::Bold] = font;
        }
        else if ( suffixName == "italic" )
        {
            fonts[numFont][( size_t )SystemFontType::Italic] = font;
        }
        else if ( suffixName == "bolditalic" )
        {
            fonts[numFont][( size_t )SystemFontType::BoldItalic] = font;
        }
    }

    return fonts;
}

} // namespace MR
