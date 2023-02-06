#include "MRSystem.h"
#include "MRStringConvert.h"
#include "MRPch/MRSpdlog.h"
#include "MRConfig.h"
#include <cstring>
#include <filesystem>
#include <fstream>

#ifdef _WIN32

#ifndef _MSC_VER
#include <cpuid.h>
#endif
#ifdef __MINGW32__
#include <windows.h>
#endif

#else

#if defined(__APPLE__)
#include <sys/sysctl.h>
#ifndef MRMESH_NO_CLIPBOARD
#include <clip/clip.h>
#endif
#else
#include "MRPch/MRWasm.h"
#ifndef __EMSCRIPTEN__
#include <cpuid.h>
#ifndef MRMESH_NO_CLIPBOARD
#include <clip/clip.h>
#endif
#endif
#endif
#include <pthread.h>
#include <libgen.h>
#include <unistd.h>
#include <limits.h>
#include <pwd.h>

#endif

#if defined(__APPLE__) && defined(__MACH__)
     #include <mach-o/dyld.h>
#endif

#ifndef MR_PROJECT_NAME
#define MR_PROJECT_NAME "MeshInspector"
#endif

namespace MR
{

void SetCurrentThreadName( const char * name )
{
#ifdef _MSC_VER
    const DWORD MS_VC_EXCEPTION = 0x406D1388;

    #pragma pack(push,8)
    typedef struct tagTHREADNAME_INFO
    {
        DWORD dwType; // Must be 0x1000.
        LPCSTR szName; // Pointer to name (in user addr space).
        DWORD dwThreadID; // Thread ID (-1=caller thread).
        DWORD dwFlags; // Reserved for future use, must be zero.
    } THREADNAME_INFO;
    #pragma pack(pop)

    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = name;
    info.dwThreadID = GetCurrentThreadId();
    info.dwFlags = 0;

    #pragma warning(push)
    #pragma warning(disable: 6320 6322)
    __try
    {
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
    }
    #pragma warning(pop)
#elif defined(__APPLE__) && defined(__MACH__)
    pthread_setname_np(name);
#else
#ifndef __EMSCRIPTEN__
    pthread_setname_np( pthread_self(), name);
#endif
#endif
}


std::filesystem::path GetExeDirectory()
{
#ifdef _WIN32
    HMODULE hm = GetModuleHandleA( "MRMesh.dll" );
    wchar_t szPath[MAX_PATH];
    GetModuleFileNameW( hm, szPath, MAX_PATH );
#else
    #ifdef __EMSCRIPTEN__
        return "/";
    #endif
    char szPath[PATH_MAX];
    #ifdef __APPLE__
          uint32_t size = PATH_MAX + 1;

          if (_NSGetExecutablePath(szPath, &size) != 0) {
            // Buffer size is too small.
            spdlog::error( "Executable directory is too long" );
            return {};
          }
          szPath[size] = '\0';
    #else
        ssize_t count = readlink( "/proc/self/exe", szPath, PATH_MAX );
        if( count < 0 )
        {
            spdlog::error( "Executable directory was not found" );
            return {};
        }
        if( count >= PATH_MAX )
        {
            spdlog::error( "Executable directory is too long" );
            return {};
        }
        szPath[count] = '\0';
    #endif
#endif
    auto res = std::filesystem::path{ szPath }.parent_path() / "";
    return res;
}

std::filesystem::path GetResourcesDirectory()
{
    auto exePath = GetExeDirectory();
#if defined(_WIN32) || defined(__EMSCRIPTEN__)
    return exePath;
#else
    // "build" in path means that MeshInspector is not installed to system
    // so all resources are near executable file
    if ( std::find( exePath.begin(), exePath.end(), "build" ) != exePath.end() )
        return exePath;
    #ifdef __APPLE__
        #ifdef MR_FRAMEWORK
    return "/Library/Frameworks/" + std::string( MR_PROJECT_NAME ) + ".framework/Versions/Current/Resources/";
        #else
    return "/Applications/" + std::string( MR_PROJECT_NAME ) + ".app/Contents/Resources/";
        #endif
    #else
    return "/usr/local/etc/" + std::string( MR_PROJECT_NAME ) + "/";
    #endif
#endif
}

std::filesystem::path GetFontsDirectory()
{
    auto exePath = GetExeDirectory();
#if defined(_WIN32) || defined(__EMSCRIPTEN__)
    return exePath;
#else
    // "build" in path means that MeshInspector is not installed to system
    // so all fonts are near executable file
    if ( std::find( exePath.begin(), exePath.end(), "build" ) != exePath.end() )
        return exePath;
    #ifdef __APPLE__
    return GetResourcesDirectory() / "fonts/";
    #else
    return "/usr/local/share/fonts/";
    #endif
#endif
}

std::filesystem::path GetLibsDirectory()
{
    auto exePath = GetExeDirectory();
#if defined(_WIN32) || defined(__EMSCRIPTEN__)
    return exePath;
#else
    // "build" in path means that MeshInspector is not installed to system
    // so all libs are near executable file
    if ( std::find( exePath.begin(), exePath.end(), "build" ) != exePath.end() )
        return exePath;
    #ifdef __APPLE__
        #ifdef MR_FRAMEWORK
    return "/Library/Frameworks/" + std::string( MR_PROJECT_NAME ) + ".framework/Versions/Current/lib/";
        #else
    return "/Applications/" + std::string( MR_PROJECT_NAME ) + ".app/Contents/libs/";
        #endif
    #else
    return "/usr/local/lib/" + std::string( MR_PROJECT_NAME ) + "/";
    #endif
#endif
}

std::filesystem::path getUserConfigDir()
{
#ifdef _WIN32
    std::filesystem::path filepath( _wgetenv( L"APPDATA" ) );
#else
    struct passwd* pw = getpwuid( getuid() );
    if ( !pw )
    {
        spdlog::error( "getpwuid error! errno: {}", errno );
    }
    std::filesystem::path filepath( pw->pw_dir );
    filepath /= ".local";
    filepath /= "share";
#endif
    filepath /= std::string( Config::instance().getAppName() );
    std::error_code ec;
    if ( !std::filesystem::is_directory( filepath, ec ) )
    {
        if ( ec )
            spdlog::warn( "{} {}", MR::asString( MR::systemToUtf8( ec.message().c_str() ) ), utf8string( filepath ) );
        std::filesystem::create_directories( filepath, ec );
    }
    if ( ec )
        spdlog::error( "{} {}", MR::asString( MR::systemToUtf8( ec.message().c_str() ) ), utf8string( filepath ) );
    return filepath;
}

std::filesystem::path getUserConfigFilePath()
{
    std::filesystem::path filepath = getUserConfigDir();
    filepath /= "config.json";
    return filepath;
}

std::filesystem::path GetTempDirectory()
{
    std::error_code ec;
    auto res = std::filesystem::temp_directory_path( ec );
    if ( ec )
        return {};
    res /= MR_PROJECT_NAME;

    if ( !std::filesystem::is_directory( res, ec ) )
    {
        ec.clear();
        if ( !std::filesystem::create_directories( res, ec ) )
            return {};
    }

    return res;
}

std::filesystem::path GetHomeDirectory()
{
#if defined( _WIN32 )
    return _wgetenv( L"USERPROFILE" );
#elif !defined( __EMSCRIPTEN__ )
    if ( auto home = std::getenv( "HOME" ) )
        return home;
    if ( auto* pw = getpwuid( getuid() ) )
        return pw->pw_dir;
    return {};
#else
    return {};
#endif
}

std::string GetClipboardText()
{
#if defined( __EMSCRIPTEN__ )
    return "";
#elif defined( _WIN32 )
    // Try opening the clipboard
    if ( !OpenClipboard( nullptr ) )
    {
        spdlog::error( "Could not open clipboard" );
        return "";
    }

    // Get handle of clipboard object for ANSI text
    HANDLE hData = GetClipboardData( CF_TEXT );
    if ( !hData )
    {
        spdlog::error( "Could not open clipboard" );
        CloseClipboard();
        return "";
    }

    // Lock the handle to get the actual text pointer
    char* pszText = static_cast< char* >( GlobalLock( hData ) );
    if ( !pszText )
    {
        spdlog::error( "Could not open clipboard" );
        CloseClipboard();
        return "";
    }
    // Save text in a string class instance
    std::string text( pszText );

    // Release the lock
    GlobalUnlock( hData );

    // Release the clipboard
    CloseClipboard();

    return text;
#elif defined(MRMESH_NO_CLIPBOARD)
    return "";
#else
    std::string text;
    if ( !clip::get_text( text ) )
    {
        spdlog::error( "Could not open clipboard" );
        return "";
    }
    return text;
#endif
}

void SetClipboardText( const std::string& text )
{
#if defined( __EMSCRIPTEN__ )
    ( void )text;
    return;
#elif defined( _WIN32 )
    HGLOBAL hMem = GlobalAlloc( GMEM_MOVEABLE, text.size() + 1 );
    memcpy( GlobalLock( hMem ), text.c_str(), text.size() + 1 );
    GlobalUnlock( hMem );
    OpenClipboard( 0 );
    EmptyClipboard();
    SetClipboardData( CF_TEXT, hMem );
    CloseClipboard();
#elif defined( MRMESH_NO_CLIPBOARD )
    ( void )text;
    return;
#else
    if ( !clip::set_text( text ) )
        spdlog::error( "Could not set clipboard" );
#endif
}

std::string GetMRVersionString()
{
#ifndef __EMSCRIPTEN__
    auto directory = GetResourcesDirectory();
    auto versionFilePath = directory / "mr.version";
    std::error_code ec;
    std::string configPrefix = "";
#ifndef NDEBUG
    configPrefix = "Debug: ";
#endif
    if ( !std::filesystem::exists( versionFilePath, ec ) )
        return configPrefix + "Version undefined";
    std::ifstream versFile( versionFilePath );
    if ( !versFile )
        return configPrefix + "Version reading error";
    std::string version;
    versFile >> version;
    if ( !versFile )
        return configPrefix + "Version reading error";
    return configPrefix + version;
#else
    auto *jsStr = (char *)EM_ASM_PTR({
        var version = "undefined";
        if ( typeof mrVersion != "undefined" )
            version = mrVersion;
        var lengthBytes = lengthBytesUTF8( version ) + 1;
        var stringOnWasmHeap = _malloc( lengthBytes );
        stringToUTF8( version, stringOnWasmHeap, lengthBytes );
        return stringOnWasmHeap;
    });
    std::string version( jsStr );
    free( jsStr );
    return version;
#endif
}

void OpenLink( const std::string& url )
{
#ifdef _WIN32
    ShellExecuteA( NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL );
#else
#ifdef __EMSCRIPTEN__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( window.open( UTF8ToString( $0 ) ), url.c_str() );
#pragma clang diagnostic pop
#else
#ifdef __APPLE__
    auto openres = system( ( "open " + url ).c_str() );
#else
    auto openres = system( ( "xdg-open " + url ).c_str() );
#endif
    if ( openres == -1 )
    {
        spdlog::warn( "Error opening {}", url );
    }
#endif
#endif // _WIN32
}

#ifdef _WIN32
std::filesystem::path GetWindowsInstallDirectory()
{
    wchar_t szPath[MAX_PATH];
    if ( GetWindowsDirectoryW( szPath, MAX_PATH ) )
        return szPath;
    return "C:\\Windows";
}
#endif //_WIN32

std::string GetCpuId()
{
#ifdef __EMSCRIPTEN__
    return "Web Browser";
#else
    char CPUBrandString[0x40] = {};
#if defined(__APPLE__)
    size_t size = sizeof(CPUBrandString);
    if (sysctlbyname("machdep.cpu.brand_string", &CPUBrandString, &size, NULL, 0) < 0)
        spdlog::error("Apple sysctlbyname failed!");
    return CPUBrandString;
#else
    // https://stackoverflow.com/questions/850774/how-to-determine-the-hardware-cpu-and-ram-on-a-machine
    int CPUInfo[4] = {-1};
    unsigned   nExIds, i = 0;
    // Get the information associated with each extended ID.
#ifdef _MSC_VER
    __cpuid( CPUInfo, 0x80000000 );
#else
    __cpuid( 0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3] );
#endif
    nExIds = CPUInfo[0];
    for ( i = 0x80000000; i <= nExIds; ++i )
    {
#ifdef _MSC_VER
        __cpuid( CPUInfo, i );
#else
        __cpuid( i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3] );
#endif
        // Interpret CPU brand string
        if ( i == 0x80000002 )
            std::memcpy( CPUBrandString, CPUInfo, sizeof( CPUInfo ) );
        else if ( i == 0x80000003 )
            std::memcpy( CPUBrandString + 16, CPUInfo, sizeof( CPUInfo ) );
        else if ( i == 0x80000004 )
            std::memcpy( CPUBrandString + 32, CPUInfo, sizeof( CPUInfo ) );
    }
#endif
    auto res = std::string( CPUBrandString );
    return res.substr( res.find_first_not_of(' ') );
#endif
}

std::string getOSNoSpaces()
{
    #ifdef _WIN32
    return "Windows";
    #else
    // get platform from cmake variables
    std::string platform = MR_PLATFORM;
    std::replace(platform.begin(), platform.end(), ' ', '_');
    return platform;
    #endif
}

} //namespace MR
