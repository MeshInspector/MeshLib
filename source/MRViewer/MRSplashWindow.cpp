#include "MRSplashWindow.h"
#include "MRMesh/MRSystem.h"
#include "MRGLMacro.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRImageLoad.h"
#include "MRMesh/MRSystemPath.h"
#include "MRImGuiImage.h"
#include "MRPch/MRSpdlog.h"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include "imgui_fonts_droid_sans.h"
#include "MRGladGlfw.h"
#include "ImGuiHelpers.h"

#include "MRIOExtras/MRPng.h"

namespace MR
{

SplashWindow::SplashWindow( std::string name ) :
    name_{ std::move( name ) }
{
}

SplashWindow::~SplashWindow()
{
    assert( !thread_.joinable() );
}

void SplashWindow::start()
{
    thread_ = std::thread( [&] ()
    {
        spdlog::info( "Splash window thread started" );
        SetCurrentThreadName( "Splash window thread" );

        glfwDefaultWindowHints();
        // setup window before creation
        setup_();

        // init window and imgui
        window_ = glfwCreateWindow( 1, 1, name_.c_str(), nullptr, nullptr );
        if ( !window_ )
        {
            spdlog::warn( "Failed creating splash window" );
            return;
        }

        glfwMakeContextCurrent( window_ );
        if ( !loadGL() )
        {
            spdlog::warn( "Failed load OpenGL for splash window" );
            glfwDestroyWindow( window_ );
            return;
        }

        guiContext_ = ImGui::CreateContext();
        ImGui::SetCurrentContext( guiContext_ );
        ImGui::GetIO().IniFilename = nullptr;

        if ( !ImGui_ImplGlfw_InitForOpenGL( window_, true ) )
        {
            spdlog::warn( "Failed to initialize Dear ImGui" );
            glfwDestroyWindow( window_ );
            ImGui::DestroyContext( guiContext_ );
            return;
        }
        if ( !ImGui_ImplOpenGL3_Init( "#version 150" ) )
        {
            spdlog::warn( "Failed to initialize OpenGL for Dear ImGui" );
            glfwDestroyWindow( window_ );
            ImGui::DestroyContext( guiContext_ );
            ImGui_ImplGlfw_Shutdown();
            return;
        }

        // load something
        postInit_();

        float xscale = 1.0f, yscale = 1.0f;
#ifndef __EMSCRIPTEN__
        glfwGetWindowContentScale( window_, &xscale, &yscale );
#endif
        float storedScaling = 0.5f * ( xscale + yscale );

        // resize and reposition window if needed
        positioning_( storedScaling );

        int bufSize[2];
        int winSize[2];
        glfwGetFramebufferSize( window_, &bufSize[0], &bufSize[1] );
        glfwGetWindowSize( window_, &winSize[0], &winSize[1] );
        float pixelRatio = ( float )bufSize[0] / ( float )winSize[0];
                
        reloadFont_( storedScaling, pixelRatio );

        // event loop
        while ( !terminate_ )
        {
            glfwWaitEventsTimeout( 1.0 / 60.0 ); // 60 - minFPS

            // check scaling
            glfwGetFramebufferSize( window_, &bufSize[0], &bufSize[1] );
#ifndef __EMSCRIPTEN__
            glfwGetWindowContentScale( window_, &xscale, &yscale );
#endif
            float scaling = 0.5f * ( xscale + yscale );
            if ( scaling != storedScaling )
            {
                positioning_( scaling );
                glfwGetWindowSize( window_, &winSize[0], &winSize[1] );
                pixelRatio = ( float )bufSize[0] / ( float )winSize[0];
                storedScaling = scaling;
                reloadFont_( storedScaling, pixelRatio );
                ImGui_ImplOpenGL3_DestroyDeviceObjects();
            }

            // begin frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if ( !frame_( storedScaling / pixelRatio ) )
                terminate_ = true;

            // end frame
            ImGui::Render();
            GL_EXEC( glViewport( 0, 0, bufSize[0], bufSize[1] ) );
            GL_EXEC( glClearColor( 0, 0, 0, 0 ) );
            GL_EXEC( glClear( GL_COLOR_BUFFER_BIT ) );
            ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
            glfwSwapBuffers( window_ );
        }

        preDestruct_();

        ImGui_ImplGlfw_Shutdown();
        ImGui_ImplOpenGL3_Shutdown();
        ImGui::DestroyContext( guiContext_ );
        glfwDestroyWindow( window_ );

        postDestruct_();
        spdlog::info( "Splash window thread finished" );
    } );
    afterStart_();
}

void SplashWindow::stop()
{
    if ( thread_.joinable() )
    {
        beforeStop_();
        terminate_ = true;
        thread_.join();
    }
}

#ifndef __EMSCRIPTEN__

DefaultSplashWindow::DefaultSplashWindow() :
    SplashWindow( "MeshInspector Splash" )
{
}

void DefaultSplashWindow::setup_() const
{
    glfwWindowHint( GLFW_SAMPLES, 8 );
    glfwWindowHint( GLFW_TRANSPARENT_FRAMEBUFFER, true );
    glfwWindowHint( GLFW_DECORATED, false );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
}

void DefaultSplashWindow::postInit_()
{
    auto imgRes = ImageLoad::fromPng( SystemPath::getResourcesDirectory() / "MRSplash.png" );
    if ( !imgRes )
    {
        spdlog::error( "No splash image found" );
        assert( false );
        return;
    }
    splashImage_ = std::make_shared<ImGuiImage>();
    splashImage_->update( { imgRes.value(),FilterType::Linear } );

    versionStr_ = GetMRVersionString();
}

void DefaultSplashWindow::positioning_( float )
{
    assert( splashImage_ );

    int workAreaX = 0, workAreaY = 0, workAreaW = 0, workAreaH = 0;
    glfwGetMonitorWorkarea( glfwGetPrimaryMonitor(), &workAreaX, &workAreaY, &workAreaW, &workAreaH );

    int width = std::min( int( 0.6f * float( workAreaW ) ), splashImage_->getImageWidth() );
    int height = int( float( width ) * float( splashImage_->getImageHeight() ) / float( splashImage_->getImageWidth() ) );


    glfwSetWindowSize( window_, width, height );

    int frameLeft = 0, frameTop = 0;
    glfwGetWindowFrameSize(
        window_, nullptr /*sic*/, &frameTop, nullptr,
        nullptr ); // Note that we keep `frameLeft` as zero, otherwise the centering is not perfect. A bug?
    glfwSetWindowPos(
        window_, 
        workAreaX + ( workAreaW - width + frameLeft ) / 2,
        workAreaY + ( workAreaH - height + frameTop ) / 2 );
}

void DefaultSplashWindow::reloadFont_( float hdpiScale, float pixelRatio )
{
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();

    ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
        droid_sans_compressed_size, 14.0f * hdpiScale );

    io.FontGlobalScale = 1.0f / pixelRatio;
}

bool DefaultSplashWindow::frame_( float /*scaling*/ )
{
    ImGui::SetNextWindowSize( ImGui::GetIO().DisplaySize );
    ImGui::SetNextWindowPos( ImVec2( 0, 0 ) );
    ImGui::Begin( "Splash window", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove );
    auto availableSize = ImGui::GetContentRegionAvail();
    ImGui::Image( *splashImage_, availableSize );
    ImGui::SetCursorPos( ImVec2( ImGui::GetFrameHeight() * 3, availableSize.y - ImGui::GetFrameHeight() * 2 ) );
    ImGui::PushStyleColor( ImGuiCol_Text, Color( 90, 97, 105 ).getUInt32() );
    ImGui::Text( "Copyright 2024, MeshInspector/MeshLib" );
    ImGui::SameLine( availableSize.x * 0.5f + ImGui::GetFrameHeight() * 4 );
    ImGui::Text( "%s", versionStr_.c_str() );
    ImGui::PopStyleColor();
    ImGui::End();
    return true;
}

void DefaultSplashWindow::preDestruct_()
{
    splashImage_.reset();
    versionStr_.clear();
}

#endif

}