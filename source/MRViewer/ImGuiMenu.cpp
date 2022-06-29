// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2018 Jérémie Dumas <jeremie.dumas@ens-lyon.org>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
////////////////////////////////////////////////////////////////////////////////
#include <MRMesh/MRToFromEigen.h>
#include "ImGuiMenu.h"
#include "MRMeshViewer.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_fonts_droid_sans.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRColorTheme.h"
#include "MRCommandLoop.h"
#include <GLFW/glfw3.h>

////////////////////////////////////////////////////////////////////////////////
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

namespace MR
{

void ImGuiMenu::init( MR::Viewer* _viewer )
{
    ViewerPlugin::init( _viewer );
    // Setup ImGui binding
    if ( _viewer )
    {
        IMGUI_CHECKVERSION();
        if ( !context_ )
        {
            // Single global context by default, but can be overridden by the user
            static ImGuiContext* __global_context = ImGui::CreateContext();
            context_ = __global_context;
        }
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        style.FrameRounding = 5.0f;
        reload_font();

        connect( _viewer, 0, boost::signals2::connect_position::at_front );
    }
}

void ImGuiMenu::initBackend()
{
    if ( !viewer || !viewer->isGLInitialized() )
        return;

#ifdef __EMSCRIPTEN__
    const char* glsl_version = "#version 300 es";
#else
    const char* glsl_version = "#version 150";
#endif
    rescaleStyle_();
    ImGui_ImplGlfw_InitForOpenGL( viewer->window, false );
    ImGui_ImplOpenGL3_Init( glsl_version );
}

void ImGuiMenu::load_font(int font_size)
{
    ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF(droid_sans_compressed_data,
        droid_sans_compressed_size, font_size * hidpi_scaling_);
}

void ImGuiMenu::reload_font(int font_size)
{
  hidpi_scaling_ = hidpi_scaling();
  pixel_ratio_ = pixel_ratio();
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->Clear();

  load_font(font_size);

  io.FontGlobalScale = 1.0f / pixel_ratio_;
}

void ImGuiMenu::shutdown()
{
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    disconnect();
    // User is responsible for destroying context if a custom context is given
    // ImGui::DestroyContext(*context_);
}

void ImGuiMenu::preDraw_()
{
  if( pollEventsInPreDraw )
  {
     glfwPollEvents();
  }

  if ( viewer->isGLInitialized() )
  {
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
  }
  else
  {
      // needed for dear ImGui
      ImGui::GetIO().DisplaySize = ImVec2( float( viewer->window_width ), float( viewer->window_height ) );
  }
  ImGui::NewFrame();
}

void ImGuiMenu::postDraw_()
{
  draw_menu();
  if ( viewer->isGLInitialized() )
  {
      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
  }
  else
  {
      ImGui::EndFrame();
  }
}

void ImGuiMenu::postResize_( int width, int height )
{
    if ( context_ )
    {
        ImGui::GetIO().DisplaySize.x = float( width );
        ImGui::GetIO().DisplaySize.y = float( height );
    }
}

void ImGuiMenu::postRescale_( float /*x*/, float /*y*/)
{
    reload_font();
    rescaleStyle_();
    ImGui_ImplOpenGL3_DestroyDeviceObjects();
}

void ImGuiMenu::rescaleStyle_()
{
    CommandLoop::appendCommand( [&] ()
    {
        ColorTheme::resetImGuiStyle();
        ImGui::GetStyle().ScaleAllSizes( menu_scaling() );
    } );
}

// Mouse IO
bool ImGuiMenu::onMouseDown_( Viewer::MouseButton button, int modifier)
{
    ImGui_ImplGlfw_MouseButtonCallback( viewer->window, int( button ), GLFW_PRESS, modifier );
    capturedMouse_ = ImGui::GetIO().WantCaptureMouse;
    return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiMenu::onMouseUp_( Viewer::MouseButton, int )
{
    return capturedMouse_;
}

bool ImGuiMenu::onMouseMove_(int /*mouse_x*/, int /*mouse_y*/)
{
  return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiMenu::onMouseScroll_(float delta_y)
{
  ImGui_ImplGlfw_ScrollCallback(viewer->window, 0.f, delta_y);
  return ImGui::GetIO().WantCaptureMouse;
}

// Keyboard IO
bool ImGuiMenu::onCharPressed_( unsigned  key, int /*modifiers*/ )
{
    ImGui_ImplGlfw_CharCallback( nullptr, key );
    return ImGui::GetIO().WantCaptureKeyboard;
}

bool ImGuiMenu::onKeyDown_( int key, int modifiers )
{
    ImGui_ImplGlfw_KeyCallback( viewer->window, key, 0, GLFW_PRESS, modifiers );
    return ImGui::GetIO().WantCaptureKeyboard;
}

bool ImGuiMenu::onKeyUp_( int key, int modifiers )
{
    ImGui_ImplGlfw_KeyCallback( viewer->window, key, 0, GLFW_RELEASE, modifiers );
    return ImGui::GetIO().WantCaptureKeyboard;
}

bool ImGuiMenu::onKeyRepeat_( int key, int modifiers )
{
    ImGui_ImplGlfw_KeyCallback( viewer->window, key, 0, GLFW_REPEAT, modifiers );
    return ImGui::GetIO().WantCaptureKeyboard;
}

// Draw menu
void ImGuiMenu::draw_menu()
{
  // Text labels
  draw_labels_window();

  // Viewer settings
  if (callback_draw_viewer_window) { callback_draw_viewer_window(); }
  else { draw_viewer_window(); }

  // Other windows
  if (callback_draw_custom_window) { callback_draw_custom_window(); }
  else { draw_custom_window(); }
}

void ImGuiMenu::draw_viewer_window()
{
  float menu_width = 180.f * menu_scaling();
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSizeConstraints(ImVec2(menu_width, -1.0f), ImVec2(menu_width, -1.0f));
  ImGui::Begin(
      "Viewer", nullptr,
      ImGuiWindowFlags_NoSavedSettings
      | ImGuiWindowFlags_AlwaysAutoResize
  );
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);
  if (callback_draw_viewer_menu) { callback_draw_viewer_menu(); }
  ImGui::PopItemWidth();
  ImGui::End();
}

void ImGuiMenu::draw_labels_window()
{
  // Text labels
  ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
  bool visible = true;
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0,0,0,0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
  ImGui::Begin("ViewerLabels", &visible,
      ImGuiWindowFlags_NoTitleBar
      | ImGuiWindowFlags_NoResize
      | ImGuiWindowFlags_NoMove
      | ImGuiWindowFlags_NoScrollbar
      | ImGuiWindowFlags_NoScrollWithMouse
      | ImGuiWindowFlags_NoCollapse
      | ImGuiWindowFlags_NoSavedSettings
      | ImGuiWindowFlags_NoInputs);
  for ( const auto& data : getAllObjectsInTree<const VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any ) )
  {
      draw_labels( *data );
  }

  // separate block for basis axes
  for ( const auto& viewport : viewer->viewport_list )
  {
      if ( !viewer->globalBasisAxes->isVisible( viewport.id ) )
          continue;
      if ( !viewer->globalBasisAxes->getVisualizeProperty( VisualizeMaskType::Labels, viewport.id ) )
          continue;
      for ( const auto& label : viewer->globalBasisAxes->getLabels() )
          draw_text( viewport, viewport.getParameters().globalBasisAxesXf( label.position ), Vector3f(), label.text, viewer->globalBasisAxes->getLabelsColor(), true );
  }
  for ( const auto& viewport : viewer->viewport_list )
  {
      if ( !viewer->basisAxes->isVisible( viewport.id ) )
          continue;
      if ( !viewer->basisAxes->getVisualizeProperty( VisualizeMaskType::Labels, viewport.id ) )
          continue;
      for ( const auto& label : viewer->basisAxes->getLabels() )
          draw_text( viewport, viewport.getParameters().basisAxesXf( label.position ), Vector3f(), label.text, viewer->basisAxes->getLabelsColor(), true, true );
  }
  ImGui::End();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();
}

void ImGuiMenu::draw_labels( const VisualObject& obj )
{
    const auto& labels = obj.getLabels();
    for ( const auto& viewport : viewer->viewport_list )
    {
        if ( !obj.isVisible( viewport.id ) )
            continue;
        AffineXf3f xf = obj.worldXf();
        bool clip = obj.getVisualizeProperty( VisualizeMaskType::CropLabelsByViewportRect, viewport.id );
        if ( obj.getVisualizeProperty( VisualizeMaskType::Labels, viewport.id ) )
            for ( int i = 0; i < labels.size(); ++i )
                draw_text(
                    viewport,
                    xf( labels[i].position ),
                    Vector3f( 0.0f, 0.0f, 0.0f ),
                    labels[i].text,
                    obj.getLabelsColor(),
                    clip );
        if ( obj.getVisualizeProperty( VisualizeMaskType::Name, viewport.id ) )
            draw_text(
                viewport,
                xf( obj.getBoundingBox().center() ),
                Vector3f( 0.0f, 0.0f, 0.0f ),
                obj.name(),
                obj.getLabelsColor(),
                clip );
    }

}

void ImGuiMenu::draw_text(
    const Viewport& viewport,
    const Vector3f& posOriginal,
    const Vector3f& normal,
    const std::string& text,
    const Color& color,
    bool clipByViewport,
    bool useStaticMatrix )
{
  Vector3f pos = posOriginal;
  pos += normal * 0.005f * viewport.getParameters().objectScale;
  const auto& viewportRect = viewport.getViewportRect();
  Vector3f coord = viewport.clipSpaceToViewportSpace( useStaticMatrix ? viewport.projectStaticToClipSpace( pos ) : viewport.projectToClipSpace( pos ) );
  auto viewerCoord = viewer->viewportToScreen( coord, viewport.id );

  // Draw text labels slightly bigger than normal text
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  ImVec4 clipRect( viewportRect.x, 
                   viewer->window_height - ( viewportRect.y + height( viewportRect ) ),
                   viewportRect.x + width( viewportRect ),
                   viewer->window_height - viewportRect.y );
  drawList->AddText( ImGui::GetFont(), ImGui::GetFontSize() * 1.2f,
                     ImVec2( viewerCoord.x / pixel_ratio_, viewerCoord.y / pixel_ratio_ ),
                     color.getUInt32(),
                     &text[0], &text[0] + text.size(), 0.0f,
                     clipByViewport ? &clipRect : nullptr );
}

float ImGuiMenu::pixel_ratio()
{
#if defined(__APPLE__)
    return 1.0f;
#endif
    // Computes pixel ratio for hidpi devices
    int buf_size[2];
    int win_size[2];
    GLFWwindow* window = glfwGetCurrentContext();
    if ( window )
    {
        glfwGetFramebufferSize( window, &buf_size[0], &buf_size[1] );
        glfwGetWindowSize( window, &win_size[0], &win_size[1] );
        return ( float )buf_size[0] / ( float )win_size[0];
    }
    return 1.0f;
}

float ImGuiMenu::hidpi_scaling()
{
#if defined(__APPLE__)
    return 1.0f;
#endif
    // Computes scaling factor for hidpi devices
    float xscale{ 1.0f }, yscale{ 1.0f };
#ifndef __EMSCRIPTEN__
    GLFWwindow* window = glfwGetCurrentContext();
    if ( window )
    {
        glfwGetWindowContentScale( window, &xscale, &yscale );
    }
#endif
    return 0.5f * ( xscale + yscale );
}

float ImGuiMenu::menu_scaling() const
{
#ifdef __EMSCRIPTEN__
    return float( emscripten_get_device_pixel_ratio() );
#else
    return hidpi_scaling_ / pixel_ratio_;
#endif
}

ImGuiContext* ImGuiMenu::getCurrentContext() const
{
    return context_;
}

} // end namespace
