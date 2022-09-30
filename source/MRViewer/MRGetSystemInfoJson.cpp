#include "MRGetSystemInfoJson.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include "MRMesh/MRSystem.h"
#include "MRViewer/MRGLMacro.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRSpdlog.h"
#include "MRGladGlfw.h"
#ifdef _WIN32
#include <shlobj.h>
#include <windows.h>
#include <psapi.h>
#endif

namespace MR
{

Json::Value GetSystemInfoJson()
{
    Json::Value root;
    root["Version"] = GetMRVersionString();
    auto& cpuInfo = root["CPU Info"];
    cpuInfo["CPU"] = GetCpuId();
    cpuInfo["Concurrent threads"] = ( Json::UInt64 )tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism );
    if ( getViewerInstance().isGLInitialized() )
    {
        auto& glInfo = root["OpenGL Info"];
        GL_EXEC();
        glInfo["Vendor"] = std::string( ( const char* )glGetString( GL_VENDOR ) );
        GL_EXEC();
        glInfo["Renderer"] = std::string( ( const char* )glGetString( GL_RENDERER ) );
        GL_EXEC();
        glInfo["OpenGL Version"] = std::string( ( const char* )glGetString( GL_VERSION ) );
        GL_EXEC();
    }

    int frameBufferSizeX, frameBufferSizeY;
    int windowSizeX, windowSizeY;
    glfwGetFramebufferSize( getViewerInstance().window, &frameBufferSizeX, &frameBufferSizeY );
    glfwGetWindowSize( getViewerInstance().window, &windowSizeX, &windowSizeY );
    auto& windowInfo = root["Window Info"];
    windowInfo["Framebuffer size"] = fmt::format( "{} x {}", frameBufferSizeX, frameBufferSizeY );
    windowInfo["Window size"] = fmt::format( "{} x {}", windowSizeX, windowSizeY );
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
#endif
    return root;
}

}