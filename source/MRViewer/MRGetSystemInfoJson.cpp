#include "MRGetSystemInfoJson.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include "MRMesh/MRSystem.h"
#include "MRGLMacro.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRSpdlog.h"
#include "MRGladGlfw.h"
#include "MRCudaAccessor.h"
#ifdef _WIN32
#include <shlobj.h>
#include <windows.h>
#include <psapi.h>
#else
#ifndef __APPLE__
#include <sys/sysinfo.h>
#endif
#endif

namespace MR
{

Json::Value GetSystemInfoJson()
{
    Json::Value root;
    root["Version"] = GetMRVersionString();
    root["OS Version"] = GetDetailedOSName();
    auto& cpuInfo = root["CPU Info"];
    cpuInfo["CPU"] = GetCpuId();
    cpuInfo["Concurrent threads"] = ( Json::UInt64 )tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism );
    auto& windowInfo = root["Window Info"];
    if ( getViewerInstance().isGLInitialized() )
    {
        auto& glInfo = root["Graphics Info"];
        GL_EXEC();
        glInfo["OpenGL Vendor"] = std::string( ( const char* )glGetString( GL_VENDOR ) );
        GL_EXEC();
        glInfo["OpenGL Renderer"] = std::string( ( const char* )glGetString( GL_RENDERER ) );
        GL_EXEC();
        glInfo["OpenGL Version"] = std::string( ( const char* )glGetString( GL_VERSION ) );
        GL_EXEC();

        glInfo["CUDA memory"] = CudaAccessor::isCudaAvailable() ?
            fmt::format( "{:.1f} GB", CudaAccessor::getCudaFreeMemory() / 1024 / 1024 / 1024.0f ) :
            "n/a";

        int frameBufferSizeX, frameBufferSizeY;
        int windowSizeX, windowSizeY;
        glfwGetFramebufferSize( getViewerInstance().window, &frameBufferSizeX, &frameBufferSizeY );
        glfwGetWindowSize( getViewerInstance().window, &windowSizeX, &windowSizeY );
        windowInfo["Framebuffer size"] = fmt::format( "{} x {}", frameBufferSizeX, frameBufferSizeY );
        windowInfo["Window size"] = fmt::format( "{} x {}", windowSizeX, windowSizeY );
    }
    else
    {
        windowInfo["Mode"] = "No Window mode";
    }
    if ( auto menu = getViewerInstance().getMenuPlugin() )
    {
        windowInfo["Pixel ratio"] = fmt::format( "{}", menu->pixel_ratio() );
        windowInfo["System scaling"] = fmt::format( "{}", menu->hidpi_scaling() );
        windowInfo["Menu scaling"] = fmt::format( "{}", menu->menu_scaling() );
    }
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof( MEMORYSTATUSEX );
    GlobalMemoryStatusEx( &memInfo );
    auto& memoryInfo = root["Memory Info"];
    memoryInfo["Virtual memory total"] = fmt::format( "{:.1f} GB", memInfo.ullTotalPageFile / 1024 / 1024 / 1024.0f );
    memoryInfo["Virtual memory available"] = fmt::format( "{:.1f} GB", memInfo.ullAvailPageFile / 1024 / 1024 / 1024.0f );

    memoryInfo["Physical memory total"] = fmt::format( "{:.1f} GB", memInfo.ullTotalPhys / 1024 / 1024 / 1024.0f );
    memoryInfo["Physical memory available"] = fmt::format( "{:.1f} GB", memInfo.ullAvailPhys / 1024 / 1024 / 1024.0f );
    memoryInfo["Physical memory total MB"] = std::to_string( memInfo.ullTotalPhys / 1024 / 1024 );
#else
#ifndef __EMSCRIPTEN__
    // if lunix
#ifndef __APPLE__
    struct sysinfo sysInfo;
    if ( sysinfo( &sysInfo ) == 0 )
    {
        auto& memoryInfo = root["Memory Info"];
        memoryInfo["Physical memory total"] = fmt::format( "{:.1f} GB", sysInfo.totalram * sysInfo.mem_unit / 1024 / 1024 / 1024.0f );
        memoryInfo["Physical memory available"] = fmt::format( "{:.1f} GB", sysInfo.freeram * sysInfo.mem_unit / 1024 / 1024 / 1024.0f );
        memoryInfo["Physical memory total MB"] = std::to_string( sysInfo.totalram * sysInfo.mem_unit / 1024 / 1024 );
    }
#else // if apple
    char buf[1024];
    unsigned buflen = 0;
    char line[256];
    FILE* sw_vers = popen( "sysctl hw.memsize", "r" );
    while ( fgets( line, sizeof( line ), sw_vers ) != NULL )
    {
        int l = snprintf( buf + buflen, sizeof( buf ) - buflen, "%s", line );
        buflen += l;
        assert( buflen < sizeof( buf ) );
    }
    pclose( sw_vers );
    auto aplStr = std::string( buf );
    auto memPos = aplStr.find( ": " );
    if ( memPos != std::string::npos )
    {
        auto aplMem = std::atoll( aplStr.c_str() + memPos + 2 );
        if ( aplMem != 0 )
        {
            auto& memoryInfo = root["Memory Info"];
            memoryInfo["Physical memory total"] = fmt::format( "{:.1f} GB", aplMem / 1024 / 1024 / 1024.0f );
            memoryInfo["Physical memory total MB"] = std::to_string( aplMem / 1024 / 1024 );
        }
    }
#endif
#endif
#endif
    return root;
}

}