#include "MRGetSystemInfoJson.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include "MRGLMacro.h"
#include "MRGladGlfw.h"
#include "MRCudaAccessor.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRSpdlog.h"
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
        int curSamples = 0;
        GL_EXEC( glGetIntegerv( GL_SAMPLES, &curSamples ) );
        
        glInfo["MSAA"] = std::to_string( curSamples );

        glInfo["CUDA memory"] = CudaAccessor::isCudaAvailable() ?
            bytesString( CudaAccessor::getCudaFreeMemory() ) :
            "n/a";

        int cudaRTVersion = CudaAccessor::getCudaRuntimeVersion();
        int cudaMaxDriverVersion = CudaAccessor::getCudaMaxDriverSupportedVersion();
        glInfo["CUDA Versions"] = cudaRTVersion != 0 && cudaMaxDriverVersion != 0 ? 
            fmt::format( "{}.{}/{}.{}", cudaRTVersion / 1000, ( cudaRTVersion % 1000 ) / 10, cudaMaxDriverVersion / 1000, ( cudaMaxDriverVersion % 1000 ) / 10 )
            : "n/a";

        int cudaCCMajor = CudaAccessor::getComputeCapabilityMajor();
        int cudaCCMinor = CudaAccessor::getComputeCapabilityMinor();
        glInfo["CUDA Compute Capability"] = cudaCCMajor != 0 && cudaCCMinor != 0 ? fmt::format( "{}.{}", cudaCCMajor, cudaCCMinor ) : "n/a";

        int frameBufferSizeX, frameBufferSizeY;
        int windowSizeX, windowSizeY;
        glfwGetFramebufferSize( getViewerInstance().window, &frameBufferSizeX, &frameBufferSizeY );
        glfwGetWindowSize( getViewerInstance().window, &windowSizeX, &windowSizeY );
        windowInfo["Framebuffer size"] = fmt::format( "{} x {}", frameBufferSizeX, frameBufferSizeY );
        windowInfo["Window size"] = fmt::format( "{} x {}", windowSizeX, windowSizeY );


        int count;
        auto monitors = glfwGetMonitors( &count );
        int maxWidth = 0, maxHeight = 0, maxScale = 0;
        for ( int i = 0; i < count; ++i )
        {
            const GLFWvidmode* mode = glfwGetVideoMode( monitors[i] );
            if ( mode && mode->width > maxWidth )
            {
                maxWidth = mode->width;
                maxHeight = mode->height;
                float xScale, yScale;
                glfwGetMonitorContentScale( monitors[i], &xScale, &yScale );
                maxScale = int( xScale * 100 );
            }
        }
        if ( maxWidth > 0 && maxHeight > 0 && maxScale > 0 )
        {
            auto& monitorInfo = windowInfo["BiggestMonitor"];
            monitorInfo["Width"] = maxWidth;
            monitorInfo["Height"] = maxHeight;
            monitorInfo["ScalingPercent"] = maxScale;
        }
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
    memoryInfo["Virtual memory total"] = bytesString( memInfo.ullTotalPageFile );
    memoryInfo["Virtual memory available"] = bytesString( memInfo.ullAvailPageFile );

    memoryInfo["Physical memory total"] = bytesString( memInfo.ullTotalPhys );
    memoryInfo["Physical memory available"] = bytesString( memInfo.ullAvailPhys );
    memoryInfo["Physical memory total MB"] = std::to_string( memInfo.ullTotalPhys / 1024 / 1024 );

    const auto procMem = getProccessMemoryInfo();
    auto& pm = root["Process Memory"];
    pm["Peak virtual memory"] = bytesString( procMem.maxVirtual );
    pm["Current virtual memory"] = bytesString( procMem.currVirtual );
    pm["Peak physical memory"] = bytesString( procMem.maxPhysical );
    pm["Current physical memory"] = bytesString( procMem.currPhysical );
#else
    const auto physMem = getPhysicalMemoryTotal();
    if ( physMem > 0 )
    {
    }
#ifndef __EMSCRIPTEN__
    // if linux
#ifndef __APPLE__
    struct sysinfo sysInfo;
    if ( sysinfo( &sysInfo ) == 0 )
    {
        auto& memoryInfo = root["Memory Info"];
        memoryInfo["Physical memory total"] = bytesString( sysInfo.totalram * sysInfo.mem_unit );
        memoryInfo["Physical memory available"] = bytesString( sysInfo.freeram * sysInfo.mem_unit );
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
            memoryInfo["Physical memory total"] = bytesString( aplMem );
            memoryInfo["Physical memory total MB"] = std::to_string( aplMem / 1024 / 1024 );
        }
    }
#endif
#endif
#endif
    return root;
}

}