#include "MRSystem.h"
#include "MRStringConvert.h"
#include <cstring>
#include <filesystem>
#include <fstream>

#ifdef _WIN32

#ifndef _MSC_VER
#include <cpuid.h>
#endif

#else

#include "MRPch/MRSpdlog.h"
#if defined(__APPLE__)
#include <sys/sysctl.h>
#else
#ifndef __EMSCRIPTEN__
#include <cpuid.h>
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
#define MR_PROJECT_NAME "MeshRUs"
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
#ifdef _WIN32
    return exePath;
#else
    // "build" in path means that MeshRUs is not installed to system
    // so all resources are near executable file
    if ( std::find( exePath.begin(), exePath.end(), "build" ) != exePath.end() )
        return exePath;
    #ifdef __APPLE__
    return "/Applications/" + std::string( MR_PROJECT_NAME ) + ".app/Contents/Resources/";
    #else
    return "/usr/local/etc/" + std::string( MR_PROJECT_NAME ) + "/";
    #endif
#endif
}

std::filesystem::path GetFontsDirectory()
{
    auto exePath = GetExeDirectory();
#ifdef _WIN32
    return exePath;
#else
    // "build" in path means that MeshRUs is not installed to system
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
#ifdef _WIN32
    return exePath;
#else
    // "build" in path means that MeshRUs is not installed to system
    // so all libs are near executable file
    if ( std::find( exePath.begin(), exePath.end(), "build" ) != exePath.end() )
        return exePath;
    #ifdef __APPLE__
    return "/Applications/" + std::string( MR_PROJECT_NAME ) + ".app/Contents/Frameworks/";
    #else
    return "/usr/local/lib/" + std::string( MR_PROJECT_NAME ) + "/";
    #endif
#endif
}

std::filesystem::path getUserConfigFilePath( const std::string& appName )
{
#ifdef _WIN32
    std::filesystem::path filepath( getenv( "APPDATA" ) );
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
    filepath /= appName;
    std::error_code ec;
    if ( !std::filesystem::is_directory( filepath, ec ) )
    {
        if ( ec )
            spdlog::warn( "{} {}", MR::asString( MR::systemToUtf8( ec.message().c_str() ) ), utf8string( filepath ) );
        std::filesystem::create_directories( filepath, ec );
    }
    if ( ec )
        spdlog::error( "{} {}", MR::asString( MR::systemToUtf8( ec.message().c_str() ) ), utf8string( filepath ) );
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

std::string GetMRVersionString()
{
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

#ifndef __EMSCRIPTEN__
std::string GetCpuId()
{
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
    return CPUBrandString;
}
#endif

} //namespace MR
