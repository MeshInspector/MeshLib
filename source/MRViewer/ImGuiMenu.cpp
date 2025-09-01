// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2018 Jérémie Dumas <jeremie.dumas@ens-lyon.org>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
////////////////////////////////////////////////////////////////////////////////
#include "ImGuiMenu.h"
#include "MRMesh/MRChrono.h"
#include "MRMesh/MRObjectDimensionsEnum.h"
#include "MRViewer.h"
#include "MRRecentFilesStore.h"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include "imgui_fonts_droid_sans.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRObjectMesh.h"
#include <MRMesh/MRSceneRoot.h>
#include "MRColorTheme.h"
#include "MRCommandLoop.h"
#include "MRViewport.h"
#include <GLFW/glfw3.h>
#include "MRShortcutManager.h"
#include "MRMesh/MRTimer.h"
#include "ImGuiHelpers.h"
#include "MRAppendHistory.h"
#include "MRShowModal.h"
#include "MRMesh/MRChangeNameAction.h"
#include "MRMesh/MRChangeSceneAction.h"
////////////////////////////////////////////////////////////////////////////////
#include "MRPch/MRWasm.h"
#include "MRPch/MRSuppressWarning.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRRibbonButtonDrawer.h"
#include "MRSymbolMesh/MRObjectLabel.h"

#include "MRMesh/MRChangeXfAction.h"
#include "MRMeshModifier.h"
#include "MRPch/MRSpdlog.h"
#include "MRProgressBar.h"
#include "MRFileDialog.h"
#include "MRModalDialog.h"

#include <MRMesh/MRMesh.h>
#include <MRMesh/MRObjectLoad.h>
#include <MRMesh/MRObject.h>
#include <MRMesh/MRBox.h>
#include "MRMesh/MRBitSet.h"
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRHistoryStore.h"
#include "ImGuiHelpers.h"
#include "MRAppendHistory.h"
#include "MRMesh/MRCombinedHistoryAction.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRTimer.h"
#include "MRSymbolMesh/MRChangeLabelAction.h"
#include "MRMesh/MRMatrix3Decompose.h"
#include "MRMesh/MRFeatureObject.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRMesh/MRAngleMeasurementObject.h"
#include "MRMesh/MRDistanceMeasurementObject.h"
#include "MRMesh/MRRadiusMeasurementObject.h"
#include "imgui_internal.h"
#include "MRRibbonConstants.h"
#include "MRRibbonFontManager.h"
#include "MRViewportGlobalBasis.h"
#include "MRUIStyle.h"
#include "MRRibbonSchema.h"
#include "MRRibbonMenu.h"
#include "MRMouseController.h"
#include "MRSceneCache.h"
#include "MRSceneObjectsListDrawer.h"
#include "MRUIRectAllocator.h"
#include "MRVisualObjectTag.h"
#include "MRMesh/MRSceneColors.h"

#ifndef MRVIEWER_NO_VOXELS
#include "MRVoxels/MRObjectVoxels.h"
#include "MRVoxels/MRVoxelsSave.h"
#endif

#ifndef __EMSCRIPTEN__
#include <fmt/chrono.h>
#endif

#include "MRPch/MRWinapi.h"

#include <bitset>

namespace
{
// Reserved keys block
using OrderedKeys = std::bitset<ImGuiKey_KeysData_SIZE>;

OrderedKeys& getOrderedKeys()
{
    static OrderedKeys orderedKeys;
    return orderedKeys;
}
}

namespace MR
{

const std::shared_ptr<ImGuiMenu>& ImGuiMenu::instance()
{
    return getViewerInstance().getMenuPlugin();
}

namespace
{
// translation multiplier that limits its maximum value depending on object size
constexpr float cMaxTranslationMultiplier = 0xC00;
} // namespace

constexpr std::array<const char*, size_t( MR::Viewer::EventType::Count )> cGLPrimitivesCounterNames =
{
    "Point Array Size",
    "Line Array Size",
    "Triangle Array Size",
    "Point Elements Number",
    "Line Elements Number",
    "Triangle Elements Number"
};

constexpr std::array<const char*, size_t( MR::Viewer::EventType::Count )> cEventCounterNames =
{
    "Mouse Down",
    "Mouse Up",
    "Mouse Move",
    "Mouse Scroll",
    "Key Down",
    "Key Up",
    "Key Repeat",
    "Char Pressed"
};

void selectRecursive( Object& obj )
{
    obj.select( true );
    for ( auto& child : obj.children() )
        if ( child )
            selectRecursive( *child );
}

void ImGuiMenu::init( MR::Viewer* _viewer )
{
    MR_TIMER;
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

    sceneObjectsList_ = std::make_shared<SceneObjectsListDrawer>();
    setupShortcuts_();
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

void reserveKeyEvent( ImGuiKey key )
{
    getOrderedKeys()[key] = true;
}

void ImGuiMenu::startFrame()
{
    if ( pollEventsInPreDraw )
    {
        glfwPollEvents();
    }

    // clear ordered keys
    getOrderedKeys() = {};

    if ( viewer->isGLInitialized() )
    {
        ImGui_ImplOpenGL3_NewFrame();
#ifdef __EMSCRIPTEN__
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
#endif

        ImGui_ImplGlfw_NewFrame();

#ifdef __EMSCRIPTEN__
        ImGui::GetIO().ConfigFlags &= ~( ImGuiConfigFlags_NoMouseCursorChange );
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
        EM_ASM( customSetCursor( $0 ), int( ImGui::GetMouseCursor() ) );
#pragma clang diagnostic pop
#endif

        if ( viewer->hasScaledFramebuffer() )
        {
            // we want ImGui to think it is common scaling in case of retina monitor
            ImGui::GetIO().DisplaySize = ImVec2( float( viewer->framebufferSize.x ), float( viewer->framebufferSize.y ) );
            ImGui::GetIO().DisplayFramebufferScale = ImVec2( 1, 1 );

            // ImGui takes mouse position from glfw each frame, but we scale mouse events, so need to update it
            if ( context_ )
            {
                ImGuiInputEvent e;
                auto curPos = Vector2f( viewer->mouseController().getMousePos() );
                if ( !context_->InputEventsQueue.empty() && context_->InputEventsQueue.back().Type == ImGuiInputEventType_MousePos )
                {
                    context_->InputEventsQueue.back().MousePos.PosX = curPos.x;
                    context_->InputEventsQueue.back().MousePos.PosY = curPos.y;
                }
            }
        }
    }
    else
    {
        // needed for dear ImGui
        // should be window size
        ImGui::GetIO().DisplaySize = ImVec2( float( viewer->framebufferSize.x ), float( viewer->framebufferSize.y ) );
    }
    auto& style = ImGui::GetStyle();
    if ( !needModalBgChange_ )
        style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4( 0.0f, 0.0f, 0.0f, 0.8f );
    else
    {
        if ( modalMessageType_ == NotificationType::Error )
            style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4( 1.0f, 0.2f, 0.2f, 0.5f );
        else if ( modalMessageType_ == NotificationType::Warning )
            style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4( 1.0f, 0.86f, 0.4f, 0.5f );
        else // if ( modalMessageType_ == MessageType::Info )
            style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4( 0.9f, 0.9f, 0.9f, 0.5f );

    }
    ImGui::NewFrame();
    UI::getDefaultWindowRectAllocator().invalidateClosedWindows();
}

void ImGuiMenu::finishFrame()
{
    draw_menu();
    prevFrameFocusPlugin_ = nullptr;
    if ( context_ && !context_->WindowsFocusOrder.empty() && !ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup ) )
    {
        for ( int i = context_->WindowsFocusOrder.size() - 1; i >= 0; --i )
        {
            auto* win = context_->WindowsFocusOrder[i];
            if ( win && win->Active && std::string( win->Name ).find( StateBasePlugin::UINameSuffix() ) != std::string::npos )
            {
                prevFrameFocusPlugin_ = win;
                break;
            }
        }
    }
    ProgressBar::onFrameEnd();
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

std::filesystem::path ImGuiMenu::getMenuFontPath() const
{
#ifdef _WIN32
    // get windows font
    wchar_t winDir[MAX_PATH];
    GetWindowsDirectoryW( winDir, MAX_PATH );
    std::filesystem::path winDirPath( winDir );
    winDirPath /= "Fonts";
    winDirPath /= "Consola.ttf";
    return winDirPath;
#else
    return {};
#endif
}

const ImVec4 undefined = ImVec4( 0.5f, 0.5f, 0.5f, 0.5f );



// at least one of selected is true - first,
// all selected are true - second
static std::pair<bool, bool> getRealValue( const std::vector<std::shared_ptr<MR::VisualObject>>& selected,
                                    AnyVisualizeMaskEnum type, MR::ViewportMask viewportId, bool inverseInput = false )
{
    bool atLeastOneTrue = false;
    bool allTrue = true;
    for ( const auto& data : selected )
    {
        bool isThisTrue = data && data->getVisualizeProperty( type, viewportId );
        if ( inverseInput )
            isThisTrue = !isThisTrue;
        atLeastOneTrue = atLeastOneTrue || isThisTrue;
        allTrue = allTrue && isThisTrue;
    }
    allTrue = allTrue && atLeastOneTrue;
    return { atLeastOneTrue,allTrue };
}

void ImGuiMenu::addMenuFontRanges_( ImFontGlyphRangesBuilder& builder ) const
{
    builder.AddRanges( ImGui::GetIO().Fonts->GetGlyphRangesCyrillic() );
    builder.AddChar( 0x2116 ); // NUMERO SIGN (shift+3 on cyrillic keyboards)
    builder.AddChar( 0x2212 ); // MINUS SIGN
    builder.AddChar( 0x222A ); // UNION
    builder.AddChar( 0x2229 ); // INTERSECTION
    builder.AddChar( 0x2208 ); // INSIDE
    builder.AddChar( 0x2209 ); // OUTSIDE
    // Characters not in the font, with custom glyphs added in `addCustomGlyphs_`:
    // 0x207B SUPERSCRIPT MINUS
#ifndef __EMSCRIPTEN__
    builder.AddRanges( ImGui::GetIO().Fonts->GetGlyphRangesChineseSimplifiedCommon() );
#endif
}

void ImGuiMenu::load_font(int font_size)
{
#ifdef _WIN32
    if ( viewer->isGLInitialized() )
    {
        ImGuiIO& io = ImGui::GetIO();

        auto fontPath = getMenuFontPath();

        ImVector<ImWchar> ranges;
        ImFontGlyphRangesBuilder builder;
        addMenuFontRanges_( builder );
        builder.BuildRanges( &ranges );

        if ( !io.Fonts->AddFontFromFileTTF(
            utf8string( fontPath ).c_str(), font_size * menu_scaling(),
            nullptr, ranges.Data ) )
        {
            assert( false && "Failed to load font!" );
            spdlog::error( "Failed to load font from `{}`.", utf8string( fontPath ) );

            ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
                droid_sans_compressed_size, font_size * hidpi_scaling_ );
        }
        io.Fonts->Build();
    }
    else
    {
        ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
            droid_sans_compressed_size, font_size * hidpi_scaling_ );
        ImGui::GetIO().Fonts[0].Build();
    }
#else
    ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
        droid_sans_compressed_size, font_size * hidpi_scaling_);
    //TODO: expand for non-Windows systems
#endif
}

void ImGuiMenu::reload_font(int font_size)
{
  hidpi_scaling_ = hidpi_scaling();
  pixel_ratio_ = pixel_ratio();
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->Clear();

  load_font(font_size);
}

void ImGuiMenu::shutdown()
{
    // Cleanup
    if ( viewer && viewer->isGLInitialized() )
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
    }

    disconnect();
    // User is responsible for destroying context if a custom context is given
    // ImGui::DestroyContext(*context_);

    // to release shared_ptr's on Ribbon items and let them be destroyed
    shortcutManager_.reset();
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

bool ImGuiMenu::spaceMouseMove_( const Vector3f& /*translate*/, const Vector3f& /*rotate*/ )
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::spaceMouseDown_( int /*key*/ )
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::touchpadRotateGestureBegin_()
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::touchpadRotateGestureUpdate_( float )
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::touchpadRotateGestureEnd_()
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::touchpadSwipeGestureBegin_()
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::touchpadSwipeGestureUpdate_( float, float deltaY, bool )
{
    // touchpad scroll values are larger than mouse ones
    constexpr float cTouchpadScrollCoef = 0.1f;

    if ( ImGui::GetIO().WantCaptureMouse )
    {
        // allow ImGui to process the touchpad swipe gesture as a scroll exclusively
        ImGui_ImplGlfw_ScrollCallback( viewer->window, 0.f, deltaY * cTouchpadScrollCoef );
        // do extra frames to prevent imgui calculations ping
        viewer->incrementForceRedrawFrames( viewer->forceRedrawMinimumIncrementAfterEvents, viewer->swapOnLastPostEventsRedraw );
        return true;
    }

    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::touchpadSwipeGestureEnd_()
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::touchpadZoomGestureBegin_()
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::touchpadZoomGestureUpdate_( float, bool )
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::touchpadZoomGestureEnd_()
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

void ImGuiMenu::rescaleStyle_()
{
    CommandLoop::appendCommand( [&] ()
    {
        ColorTheme::resetImGuiStyle(); // apply scaling inside
    } );
}

// Mouse IO
bool ImGuiMenu::onMouseDown_( Viewer::MouseButton button, int modifier)
{
    capturedMouse_ = ImGui::GetIO().WantCaptureMouse
        || bool( uiRenderManager_->consumedInteractions & BasicUiRenderTask::InteractionMask::mouseHover );

    ImGui_ImplGlfw_MouseButtonCallback( viewer->window, int( button ), GLFW_PRESS, modifier );

    if ( !capturedMouse_ )
    {
        auto* ctx = ImGui::GetCurrentContext();
        if ( ctx->ActiveId == ctx->TempInputId ) // some item in temp input mode, but we clicked out of imgui
            ImGui::ClearActiveID(); // disable temp input mode
    }

    return capturedMouse_;
}

bool ImGuiMenu::onMouseUp_( Viewer::MouseButton button, int modifier )
{
    ImGui_ImplGlfw_MouseButtonCallback( viewer->window, int( button ), GLFW_RELEASE, modifier );
    return capturedMouse_;
}

bool ImGuiMenu::onMouseMove_( int mouse_x, int mouse_y )
{
    ImGui_ImplGlfw_CursorPosCallback( viewer->window, double( mouse_x ), double( mouse_y ) );
    return false;
}

bool ImGuiMenu::onMouseScroll_( float delta_y )
{
    if ( ImGui::GetIO().WantCaptureMouse || bool( uiRenderManager_->consumedInteractions & BasicUiRenderTask::InteractionMask::mouseScroll ) )
    {
        // allow ImGui to process the scroll exclusively
        ImGui_ImplGlfw_ScrollCallback( viewer->window, 0.f, delta_y );
        // do extra frames to prevent imgui calculations ping
        viewer->incrementForceRedrawFrames( viewer->forceRedrawMinimumIncrementAfterEvents, viewer->swapOnLastPostEventsRedraw );
        return uiRenderManager_->canConsumeEvent( BasicUiRenderTask::InteractionMask::mouseScroll );
    }

    return false;
}

void ImGuiMenu::cursorEntrance_( [[maybe_unused]] bool entered )
{
#ifdef __EMSCRIPTEN__
    static bool isInside = false;
    if ( entered )
    {
        if ( !isInside )
        {
            ImGui::GetIO().ConfigFlags &= (~ImGuiConfigFlags_NoMouse);
            isInside = true;
        }
    }
    else
    {
        bool anyPressed =
            ImGui::IsMouseDown( ImGuiMouseButton_Left ) ||
            ImGui::IsMouseDown( ImGuiMouseButton_Right ) ||
            ImGui::IsMouseDown( ImGuiMouseButton_Middle );
        if ( !anyPressed )
        {
            ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
            isInside = false;
            EM_ASM( postEmptyEvent( 100, 2 ) );
        }
    }
#endif
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
    return ImGui::GetIO().WantCaptureKeyboard || getOrderedKeys()[GlfwToImGuiKey_Duplicate( key )];
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
  ImGui::Begin("ViewerLabels##[rect_allocator_ignore]", &visible,
      ImGuiWindowFlags_NoTitleBar
      | ImGuiWindowFlags_NoResize
      | ImGuiWindowFlags_NoMove
      | ImGuiWindowFlags_NoScrollbar
      | ImGuiWindowFlags_NoScrollWithMouse
      | ImGuiWindowFlags_NoCollapse
      | ImGuiWindowFlags_NoSavedSettings
      | ImGuiWindowFlags_NoInputs);

  ImGui::End();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();
}

void ImGuiMenu::draw_text(
    const Viewport& viewport,
    const Vector3f& posOriginal,
    const Vector3f& normal,
    const std::string& text,
    const Color& color,
    bool clipByViewport )
{
  Vector3f pos = posOriginal;
  pos += normal * 0.005f * viewport.getParameters().objectScale;
  const auto& viewportRect = viewport.getViewportRect();
  Vector3f coord = viewport.clipSpaceToViewportSpace( viewport.projectToClipSpace( pos ) );
  auto viewerCoord = viewer->viewportToScreen( coord, viewport.id );

  // Draw text labels slightly bigger than normal text
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  ImVec4 clipRect( viewportRect.min.x,
                   viewer->framebufferSize.y - ( viewportRect.min.y + height( viewportRect ) ),
                   viewportRect.min.x + width( viewportRect ),
                   viewer->framebufferSize.y - viewportRect.min.y );
  drawList->AddText( ImGui::GetFont(), ImGui::GetFontSize() * 1.2f,
                     ImVec2( viewerCoord.x / pixel_ratio_, viewerCoord.y / pixel_ratio_ ),
                     color.getUInt32(),
                     &text[0], &text[0] + text.size(), 0.0f,
                     clipByViewport ? &clipRect : nullptr );
}

float ImGuiMenu::pixel_ratio()
{
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
    return float( emscripten_get_device_pixel_ratio() ) * userScaling_;
#elif defined __APPLE__
    return pixel_ratio_ * userScaling_;
#else
    return hidpi_scaling_ / pixel_ratio_ * userScaling_;
#endif
}

void ImGuiMenu::setUserScaling( float scaling )
{
    scaling = std::clamp( scaling, 0.5f, 4.0f );
    if ( scaling == userScaling_ )
        return;
    userScaling_ = scaling;
    CommandLoop::appendCommand( [&] ()
    {
        auto scaling = menu_scaling();
        getViewerInstance().postRescale( scaling, scaling );
    } );
}

ImGuiContext* ImGuiMenu::getCurrentContext() const
{
    return context_;
}

void ImGuiMenu::draw_helpers()
{
    if ( showShortcuts_ )
    {
        drawShortcutsWindow_();
    }

    if ( showStatistics_ )
    {
        const auto style = ImGui::GetStyle();
        const float fpsWindowWidth = 300 * menu_scaling();
        int numLines = 5 + int( Viewer::EventType::Count ) + int( Viewer::GLPrimitivesType::Count ); // 5 - for: GL buffer size, prev frame time, swapped frames, total frames, fps;
        // TextHeight +1 for button, ItemSpacing +2 for separators
        const float fpsWindowHeight = ( style.WindowPadding.y * 2 +
                                        ImGui::GetTextLineHeight() * ( numLines + 2 ) +
                                        style.ItemSpacing.y * ( numLines + 3 ) +
                                        style.FramePadding.y * 4 );
        const float posX = getViewerInstance().framebufferSize.x - fpsWindowWidth;
        const float posY = getViewerInstance().framebufferSize.y - fpsWindowHeight;
        ImGui::SetNextWindowPos( ImVec2( posX, posY ), ImGuiCond_Appearing );
        ImGui::SetNextWindowSize( ImVec2( fpsWindowWidth, fpsWindowHeight ) );
        ImGui::Begin( "##FPS", nullptr, ImGuiWindowFlags_AlwaysAutoResize | //ImGuiWindowFlags_NoInputs |
                      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoFocusOnAppearing );
        for ( int i = 0; i<int( Viewer::GLPrimitivesType::Count ); ++i )
            ImGui::Text( "%s: %zu", cGLPrimitivesCounterNames[i], viewer->getLastFrameGLPrimitivesCount( Viewer::GLPrimitivesType( i ) ) );
        ImGui::Separator();
        for ( int i = 0; i<int( Viewer::EventType::Count ); ++i )
            ImGui::Text( "%s: %zu", cEventCounterNames[i], viewer->getEventsCount( Viewer::EventType( i ) ) );
        ImGui::Separator();
        auto glBufferSizeStr = bytesString( viewer->getStaticGLBufferSize() );
        ImGui::Text( "GL memory buffer: %s", glBufferSizeStr.c_str() );
        auto prevFrameTime = viewer->getPrevFrameDrawTimeMillisec();
        if ( prevFrameTime > frameTimeMillisecThreshold_ )
            ImGui::TextColored( ImVec4( 1.0f, 0.3f, 0.3f, 1.0f ), "Previous frame time: %.1f ms", prevFrameTime );
        else
            ImGui::Text( "Previous frame time: %.1f ms", prevFrameTime );
        ImGui::Text( "Total frames: %zu", viewer->getTotalFrames() );
        ImGui::Text( "Swapped frames: %zu", viewer->getSwappedFrames() );
        ImGui::Text( "FPS: %zu", viewer->getFPS() );

        if ( UI::buttonCommonSize( "Reset", Vector2f( -1, 0 ) ) )
        {
            viewer->resetAllCounters();
        }
        if ( UI::buttonCommonSize( "Print Time to Log", Vector2f( -1, 0 ) ) )
        {
            printTimingTree();
            ProgressBar::printTimingTree();
        }
        ImGui::End();
    }

    if ( showRenameModal_ )
    {
        showRenameModal_ = false;
        ImGui::OpenPopup( "Rename object" );
        popUpRenameBuffer_ = renameBuffer_;
    }

    const auto menuScaling = menu_scaling();
    ModalDialog renameDialog( "Rename object", {
        .headline = "Rename Object",
        .closeOnClickOutside = true,
    } );
    if ( renameDialog.beginPopup( menuScaling ) )
    {
        const auto& obj = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>().front();
        if ( !obj )
        {
            ImGui::CloseCurrentPopup();
        }
        if ( ImGui::IsWindowAppearing() )
            ImGui::SetKeyboardFocusHere();

        const auto& style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cInputPadding * menuScaling } );
        ImGui::SetNextItemWidth( renameDialog.windowWidth() - 2 * style.WindowPadding.x - style.ItemInnerSpacing.x - ImGui::CalcTextSize( "Name" ).x );
        UI::inputText( "Name", popUpRenameBuffer_, ImGuiInputTextFlags_AutoSelectAll );
        ImGui::PopStyleVar();

        const float btnWidth = cModalButtonWidth * menuScaling;
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * menuScaling } );
        if ( UI::button( "Ok", Vector2f( btnWidth, 0 ), ImGuiKey_Enter ) )
        {
            AppendHistory( std::make_shared<ChangeNameAction>( "Rename object from modal dialog", obj ) );
            obj->setName( popUpRenameBuffer_ );
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        ImGui::SetCursorPosX( renameDialog.windowWidth() - btnWidth - style.WindowPadding.x );
        if ( UI::button( "Cancel", Vector2f( btnWidth, 0 ), ImGuiKey_Escape ) )
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleVar();

        renameDialog.endPopup( menuScaling );
    }

    if ( showEditTag_ )
    {
        ImGui::OpenPopup( "Edit tag" );
        showEditTag_ = false;
    }

    ModalDialog editTagDialog( "Edit tag", {
        .headline = "Edit Tag",
        .closeButton = true,
        //.closeOnClickOutside = true, // FIXME: color picker closes the modal dialog on exit
    } );
    if ( editTagDialog.beginPopup( menuScaling ) )
    {
        if ( ImGui::IsWindowAppearing() )
            ImGui::SetKeyboardFocusHere();

        const auto& style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cInputPadding * menuScaling } );
        ImGui::SetNextItemWidth( editTagDialog.windowWidth() - 2 * style.WindowPadding.x - style.ItemInnerSpacing.x - ImGui::CalcTextSize( "Name" ).x );
        UI::inputText( "Name", tagEditorState_.name, ImGuiInputTextFlags_AutoSelectAll );
        ImGui::PopStyleVar();

        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cCheckboxPadding * menuScaling } );
        UI::checkbox( "Assign Color", &tagEditorState_.hasFrontColor );
        ImGui::PopStyleVar();

        if ( tagEditorState_.hasFrontColor )
        {
            ImGui::ColorEdit4( "Selected Color", (float*)&tagEditorState_.selectedColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel );
            ImGui::ColorEdit4( "Unselected Color", (float*)&tagEditorState_.unselectedColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel );
        }

        const float btnWidth = cModalButtonWidth * menuScaling;
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * menuScaling } );
        if ( UI::button( "Save", Vector2f( btnWidth, 0 ), ImGuiKey_Enter ) )
        {
            if ( tagEditorState_.name != tagEditorState_.initName )
            {
                if ( tagEditorState_.hasFrontColor )
                {
                    VisualObjectTagManager::unregisterTag( tagEditorState_.initName );
                    VisualObjectTagManager::registerTag( tagEditorState_.name, {
                        .selectedColor = tagEditorState_.selectedColor,
                        .unselectedColor = tagEditorState_.unselectedColor,
                    } );
                }

                for ( auto obj : getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
                {
                    if ( obj->tags().contains( tagEditorState_.initName ) )
                    {
                        obj->removeTag( tagEditorState_.initName );
                        obj->addTag( tagEditorState_.name );
                    }
                }
            }

            if ( tagEditorState_.hasFrontColor != tagEditorState_.initHasFrontColor )
            {
                if ( tagEditorState_.hasFrontColor )
                {
                    VisualObjectTagManager::registerTag( tagEditorState_.name, {
                        .selectedColor = tagEditorState_.selectedColor,
                        .unselectedColor = tagEditorState_.unselectedColor,
                    } );
                }
                else
                {
                    VisualObjectTagManager::unregisterTag( tagEditorState_.name );
                }
            }
            else if ( tagEditorState_.hasFrontColor )
            {
                VisualObjectTagManager::updateTag( tagEditorState_.name, {
                    .selectedColor = tagEditorState_.selectedColor,
                    .unselectedColor = tagEditorState_.unselectedColor,
                } );
            }

            if ( tagEditorState_.hasFrontColor || tagEditorState_.initHasFrontColor )
            {
                for ( auto& visObj : getAllObjectsInTree<VisualObject>( &SceneRoot::get() ) )
                    if ( visObj->tags().contains( tagEditorState_.name ) )
                        VisualObjectTagManager::update( *visObj, tagEditorState_.name );
            }

            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        ImGui::SetCursorPosX( editTagDialog.windowWidth() - btnWidth - style.WindowPadding.x );
        if ( UI::button( "Cancel", Vector2f( btnWidth, 0 ), ImGuiKey_Escape ) )
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleVar();

        editTagDialog.endPopup( menuScaling );
    }

    drawModalMessage_();
}

UiRenderManager& ImGuiMenu::getUiRenderManager()
{
    if ( !uiRenderManager_ )
        uiRenderManager_ = std::make_unique<UiRenderManagerImpl>();
    return *uiRenderManager_;
}

bool ImGuiMenu::simulateNameTagClick( Object& object, NameTagSelectionMode mode )
{
    if ( nameTagClickSignal( object, mode ) )
        return false;

    switch ( mode )
    {
    case NameTagSelectionMode::selectOne:
        {
            auto handleObject = [&]( auto& handleObject, Object& cur ) -> void
            {
                cur.select( &cur == &object );
                for ( const auto& child : cur.children() )
                    handleObject( handleObject, *child );
            };
            handleObject( handleObject, MR::SceneRoot::get() );
        }
        break;
    case NameTagSelectionMode::toggle:
        object.select( !object.isSelected() );
        break;
    }

    return true;
}

bool ImGuiMenu::anyImGuiWindowIsHovered() const
{
    return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiMenu::anyUiObjectIsHovered() const
{
    return bool( uiRenderManager_->consumedInteractions & BasicUiRenderTask::InteractionMask::mouseHover );
}

void ImGuiMenu::drawModalMessage_()
{
    ImGui::PushStyleColor( ImGuiCol_ModalWindowDimBg, ImVec4( 1, 0.125f, 0.125f, ImGui::GetStyle().Colors[ImGuiCol_ModalWindowDimBg].w ) );

    std::string title;
    if ( modalMessageType_ == NotificationType::Error )
        title = "Error";
    else if ( modalMessageType_ == NotificationType::Warning )
        title = "Warning";
    else //if ( modalMessageType_ == MessageType::Info )
        title = "Info";

    const std::string titleImGui = " " + title + "##modal";

    if ( showInfoModal_ &&
        !ImGui::IsPopupOpen( " Error##modal" ) && !ImGui::IsPopupOpen( " Warning##modal" ) && !ImGui::IsPopupOpen( " Info##modal" ) )
    {
        ImGui::OpenPopup( titleImGui.c_str() );
        showInfoModal_ = false;
    }

    const auto menuScaling = menu_scaling();
    ModalDialog modal( titleImGui, {
        .headline = title,
        .text = storedModalMessage_,
        .closeOnClickOutside = true,
    } );
    if ( modal.beginPopup( menuScaling ) )
    {
        const auto style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * menuScaling } );
        if ( UI::button( "Okay", Vector2f( -1, 0 ), ImGuiKey_Enter ) )
            ImGui::CloseCurrentPopup();
        ImGui::PopStyleVar();

        modal.endPopup( menuScaling );
        needModalBgChange_ = true;
    }
    else
    {
        needModalBgChange_ = false;
    }

    ImGui::PopStyleColor();
}

void ImGuiMenu::setDrawTimeMillisecThreshold( long long maxGoodTimeMillisec )
{
    frameTimeMillisecThreshold_ = maxGoodTimeMillisec;
}

void ImGuiMenu::showModalMessage( const std::string& msg, NotificationType msgType )
{
    if ( msgType == NotificationType::Error )
        spdlog::error( "Error Modal Dialog: {}", msg );
    else if ( msgType == NotificationType::Warning )
        spdlog::warn( "Warning Modal Dialog: {}", msg );
    else // if ( msgType == MessageType::Info )
        spdlog::info( "Info Modal Dialog: {}", msg );
    showRenameModal_ = false;
    showInfoModal_ = true;
    needModalBgChange_ = true;
    modalMessageType_ = msgType;
    ImGui::CloseCurrentPopup();
    storedModalMessage_ = msg;
    // this is needed to correctly resize modal window
    getViewerInstance().incrementForceRedrawFrames( 2, true );
}

void ImGuiMenu::setupShortcuts_()
{
    if ( !shortcutManager_ )
        shortcutManager_ = std::make_shared<ShortcutManager>();

    // connecting signals to events (KeyDown, KeyRepeat) with lowest priority
    shortcutManager_->connect( &getViewerInstance(), INT_MAX );
}

void ImGuiMenu::draw_scene_list()
{
    const auto& selectedObjs = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();
    // Define next window position + size
    ImGui::SetNextWindowPos( ImVec2( 180 * menu_scaling(), 0 ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowSize( ImVec2( 230 * menu_scaling(), 300 * menu_scaling() ), ImGuiCond_FirstUseEver );
    ImGui::Begin(
        "Scene", nullptr
    );
    sceneObjectsList_->draw( -1, menu_scaling() );

    sceneWindowPos_ = ImGui::GetWindowPos();
    sceneWindowSize_ = ImGui::GetWindowSize();
    ImGui::End();

    draw_selection_properties( selectedObjs );
}

void ImGuiMenu::draw_selection_properties( const std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    if ( !selectedObjs.empty() )
    {
        // Define next window position + size
        ImGui::SetNextWindowPos( ImVec2( sceneWindowPos_.x, sceneWindowPos_.y + sceneWindowSize_.y ) );
        ImGui::SetNextWindowSize( ImVec2( sceneWindowSize_.x, -1 ) );
        ImGui::Begin(
            "Selection Properties", nullptr,
            ImGuiWindowFlags_NoMove
        );
        draw_selection_properties_content( selectedObjs );
        ImGui::End();
    }
}

void ImGuiMenu::draw_selection_properties_content( const std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    drawSelectionInformation_();

    const auto& selectedVisualObjs = SceneCache::getAllObjects<VisualObject, ObjectSelectivityType::Selected>();
    bool allHaveVisualisation = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), [] ( const std::shared_ptr<VisualObject>& obj )
    {
        if ( !obj )
            return false;
        auto objMesh = obj->asType<ObjectMesh>();
        if ( objMesh && objMesh->mesh() )
            return true;
        auto objPoints = obj->asType<ObjectPoints>();
        if ( objPoints && objPoints->pointCloud() )
            return true;
        auto objLines = obj->asType<ObjectLines>();
        if ( objLines && objLines->polyline() )
            return true;
        return false;
    } );

    drawGeneralOptions( selectedObjs );

    if ( allHaveVisualisation && drawCollapsingHeader_( "Draw Options" ) )
    {
        auto selectedMask = calcSelectedTypesMask( selectedObjs );
        drawDrawOptionsCheckboxes( selectedVisualObjs, selectedMask );
        drawDrawOptionsColors( selectedVisualObjs );
        drawAdvancedOptions( selectedVisualObjs, selectedMask );
    }

    draw_custom_selection_properties( selectedObjs );

    drawRemoveButton( selectedObjs );


    drawTransform_();
}

float ImGuiMenu::drawSelectionInformation_()
{
    const auto& selectedObjs = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();

    auto& style = ImGui::GetStyle();

    float baseCursorScreenPos = ImGui::GetCursorScreenPos().y;
    auto resultingHeight = [&]
    {
        return ImGui::GetCursorScreenPos().y - baseCursorScreenPos;
    };

    if ( !drawCollapsingHeader_( "Information", ImGuiTreeNodeFlags_DefaultOpen ) || selectedObjs.empty() )
        return resultingHeight();

    // Points info
    size_t totalPoints = 0;
    size_t totalSelectedPoints = 0;
    bool pointsHaveNormals = false;
    // Meshes and lines info
    size_t totalFaces = 0;
    size_t totalSelectedFaces = 0;
    size_t totalVerts = 0;
    size_t totalEdges = 0;
    size_t totalSelectedEdges = 0;
    double totalVolume = 0.0;
    double totalArea = 0.;
    double totalSelectedArea = 0.;
    double totalLength = 0;
    float avgEdgeLen = 0.f;
    size_t holes = 0;
    size_t components = 0;
#ifndef MRVIEWER_NO_VOXELS
    // Voxels info
    std::optional<Vector3i> voxelDims = Vector3i{};
    std::optional<Vector3f> voxelSize = Vector3f{};
    std::optional<Box3i> voxelActiveBox = Box3i{};

    std::optional<float> voxelMinValue = FLT_MAX;
    std::optional<float> voxelIsoValue = FLT_MAX;
    std::optional<float> voxelMaxValue = FLT_MAX;

    // store shared parameter value: if all objects have identical parameter value, it will be displayed, otherwise it'll be hidden
    auto updateVoxelsInfo = [] <typename T, typename U> ( std::optional<T>& store, U&& value, T def = {} )
    {
        if ( store.has_value() )
        {
            if ( store.value() == def )
                store.emplace( std::forward<U>( value ) );
            else if ( store.value() != value )
                store.reset();
        }
    };
    auto isValidVoxelsInfo = [] <typename T> ( const std::optional<T>& store, T def = {} )
    {
        return store.has_value() && store.value() != def;
    };
#endif
    // Scene info
    Vector3f bsize;
    Vector3f wbsize;
    std::string bsizeStr;
    std::string wbsizeStr;
    selectionBbox_ = Box3f{};
    selectionWorldBox_ = {};

    for ( const auto& obj : selectedObjs )
    {
        // Scene info update
        if ( auto vObj = obj->asType<VisualObject>() )
        {
            if ( auto box = vObj->getBoundingBox(); box.valid() )
                selectionBbox_.include( box );
            if ( auto box = vObj->getWorldBox(); box.valid() )
                selectionWorldBox_.include( box );
        }

        // Typed info
        if ( auto pObj = obj->asType<ObjectPoints>() )
        {
            totalPoints += pObj->numValidPoints();
            totalSelectedPoints += pObj->numSelectedPoints();
            if ( auto pointCloud = pObj->pointCloud() )
                pointsHaveNormals |= pointCloud->hasNormals();
        }
        else if ( auto mObj = obj->asType<ObjectMesh>() )
        {
            if ( auto mesh = mObj->mesh() )
            {
                totalFaces += mesh->topology.numValidFaces();
                totalSelectedFaces += mObj->numSelectedFaces();
                totalVerts += mesh->topology.numValidVerts();
                totalEdges += mObj->numUndirectedEdges();
                totalSelectedEdges += mObj->numSelectedEdges();
                totalVolume += mObj->volume();
                totalArea += mObj->totalArea();
                totalSelectedArea += mObj->selectedArea();
                avgEdgeLen = mObj->avgEdgeLen();
                holes += mObj->numHoles();
                components += mObj->numComponents();
            }
        }
        else if ( auto lObj = obj->asType<ObjectLines>() )
        {
            if ( auto polyline = lObj->polyline() )
            {
                totalVerts += polyline->topology.numValidVerts();
                totalEdges += lObj->numUndirectedEdges();
                totalLength += polyline->totalLength();
                avgEdgeLen = lObj->avgEdgeLen();
                components += lObj->numComponents();
            }
        }
#ifndef MRVIEWER_NO_VOXELS
        else if ( auto vObj = obj->asType<ObjectVoxels>() )
        {
            updateVoxelsInfo( voxelDims, vObj->dimensions() );
            updateVoxelsInfo( voxelSize, vObj->voxelSize() );
            updateVoxelsInfo( voxelActiveBox, vObj->getActiveBounds() );
            updateVoxelsInfo( voxelMinValue, vObj->vdbVolume().min, FLT_MAX );
            updateVoxelsInfo( voxelIsoValue, vObj->getIsoValue(), FLT_MAX );
            updateVoxelsInfo( voxelMaxValue, vObj->vdbVolume().max, FLT_MAX );
        }
#endif
    }

    if ( selectionBbox_.valid() && selectionWorldBox_.valid() )
    {
        bsize = selectionBbox_.size();
        bsizeStr = fmt::format( "{:.3e} {:.3e} {:.3e}", bsize.x, bsize.y, bsize.z );
        wbsize = selectionWorldBox_.size();
        wbsizeStr = fmt::format( "{:.3e} {:.3e} {:.3e}", wbsize.x, wbsize.y, wbsize.z );
    }

    ImGui::PushStyleVar( ImGuiStyleVar_ScrollbarSize, 12.0f );
    MR_FINALLY{ ImGui::PopStyleVar(); };

    const float smallItemSpacingY = std::round( 0.25f * cDefaultItemSpacing * menu_scaling() );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, smallItemSpacingY } );
    MR_FINALLY{ ImGui::PopStyleVar(); };

    if ( selectedObjs.size() == 1 )
    {
        auto pObj = selectedObjs.front();
        auto lastRenameObj = lastRenameObj_.lock();
        if ( lastRenameObj != pObj )
        {
            renameBuffer_ = pObj->name();
            lastRenameObj_ = pObj;
        }
        if ( !UI::inputTextCentered( "Object Name", renameBuffer_, getSceneInfoItemWidth_(), ImGuiInputTextFlags_AutoSelectAll ) )
        {
            if ( renameBuffer_ == pObj->name() )
            {
                // clear the pointer to reload the name on next frame (if it was changed from outside)
                lastRenameObj_.reset();
            }
        }
        if ( ImGui::IsItemDeactivatedAfterEdit() )
        {
            AppendHistory( std::make_shared<ChangeNameAction>( "Rename object from information", pObj ) );
            pObj->setName( renameBuffer_ );
            lastRenameObj_.reset();
        }

        if ( auto pObjLabel = std::dynamic_pointer_cast< ObjectLabel >( pObj ) )
        {
            if ( pObjLabel != oldLabelParams_.obj )
            {
                oldLabelParams_.obj = pObjLabel;
                const auto& positionedText = pObjLabel->getLabel();
                oldLabelParams_.lastLabel = positionedText.text;
                oldLabelParams_.labelBuffer = oldLabelParams_.lastLabel;
            }

            ImGui::Spacing();
            ImGui::Spacing();

            if ( UI::inputText( "Label", oldLabelParams_.labelBuffer, ImGuiInputTextFlags_AutoSelectAll ) )
                pObjLabel->setLabel( { oldLabelParams_.labelBuffer, pObjLabel->getLabel().position } );
            if ( ImGui::IsItemDeactivatedAfterEdit() && oldLabelParams_.labelBuffer != oldLabelParams_.lastLabel )
            {
                pObjLabel->setLabel( { oldLabelParams_.lastLabel, pObjLabel->getLabel().position } );
                AppendHistory( std::make_shared<ChangeLabelAction>( "Change label", pObjLabel ) );
                pObjLabel->setLabel( { oldLabelParams_.labelBuffer, pObjLabel->getLabel().position } );
                oldLabelParams_.lastLabel = oldLabelParams_.labelBuffer;
            }
            else if ( !ImGui::IsItemActive() )
            {
                const auto& positionedText = pObjLabel->getLabel();
                oldLabelParams_.lastLabel = positionedText.text;
                oldLabelParams_.labelBuffer = oldLabelParams_.lastLabel;
            }
        }
        else if ( oldLabelParams_.obj )
        {
            oldLabelParams_.obj.reset();
        }
    }
    else
    {
        lastRenameObj_.reset();
    }

    // Feature object properties.
    if ( selectedObjs.size() == 1 && selectedObjs.front()->asType<FeatureObject>() )
    {
        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::PushItemWidth( getSceneInfoItemWidth_( 1 ) );
        drawFeaturePropertiesEditor_( selectedObjs.front() );
        ImGui::PopItemWidth();
    }
    else
    {
        editedFeatureObject_.reset();
    }

    // customize input text widget design
    const ImVec4 originalFrameBgColor = ImGui::GetStyleColorVec4( ImGuiCol_FrameBg );
    const float originalFrameBorderSize = ImGui::GetStyle().FrameBorderSize;
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, Vector2f { 3.f, 3.f } * menu_scaling() );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0 );
    ImGui::PushStyleColor( ImGuiCol_FrameBg, ImGui::GetStyleColorVec4( ImGuiCol_WindowBg ) );
    MR_FINALLY { ImGui::PopStyleVar( 2 ); ImGui::PopStyleColor( 1 ); };

    const float itemWidth = getSceneInfoItemWidth_( 3 ) * 3 + ImGui::GetStyle().ItemInnerSpacing.x * 2;

    // Width for half-width widgets.
    // There are two separate variables to prevent rounding from messing up the alignment.
    const float itemWidthHalf1 = std::round( ( itemWidth - ImGui::GetStyle().ItemInnerSpacing.x ) / 2 );
    const float itemWidthHalf2 = itemWidth - itemWidthHalf1 - ImGui::GetStyle().ItemInnerSpacing.x;

    auto textColor = ImGui::GetStyleColorVec4( ImGuiCol_Text );
    auto labelColor = textColor;
    textColor.w *= 0.5f;
    const ImVec4 selectedTextColor { 0.886f, 0.267f, 0.267f, 1.0f };

    auto drawPrimitivesInfo = [&] ( const char* label, size_t value, size_t selected = 0 )
    {
        if ( value )
        {
            std::string valueStr;
            if ( selected )
                valueStr = valueToString<NoUnit>( selected ) + " / ";
            valueStr += valueToString<NoUnit>( value );

            UI::inputTextCenteredReadOnly( label, valueStr, itemWidth, selected ? selectedTextColor : textColor, labelColor );
            if ( selected )
                UI::setTooltipIfHovered( "Selected / Total", menu_scaling() );
        }
    };

    auto drawUnitInfo = [&] <class Units> ( const char* label, auto&& value, Units )
    {
        ImGui::SetNextItemWidth( itemWidth );
        UI::readOnlyValue<Units>( label, value, textColor, {}, labelColor );
    };

    auto drawDimensionsVec3 = [&] <class Units> ( const char* label, auto&& value, Units )
    {
        ImGui::SetNextItemWidth( getSceneInfoItemWidth_() );
        UI::readOnlyValue<Units>( label, value, textColor, {}, labelColor );
    };

    if ( selectedObjs.size() == 1 )
    {
        UI::inputTextCenteredReadOnly( "Object Type", selectedObjs.front()->className(), itemWidth, textColor, labelColor );
    }
    else if ( selectedObjs.size() > 1 )
    {
        drawPrimitivesInfo( "Objects", selectedObjs.size() );
    }

    drawTagInformation_( selectedObjs, {
        .textColor = textColor,
        .labelColor = labelColor,
        .selectedTextColor = selectedTextColor,
        .itemWidth = itemWidth,
        .item2Width = getSceneInfoItemWidth_( 2 ),
        .item3Width = getSceneInfoItemWidth_( 3 ),
    } );

    // Bounding box.
    if ( selectionBbox_.valid() && !( selectedObjs.size() == 1 && selectedObjs.front()->asType<FeatureObject>() ) )
    {
        ImGui::Spacing();
        ImGui::Spacing();

        drawDimensionsVec3( "Box Size", bsize, LengthUnit{} );
        drawDimensionsVec3( "Box Min", selectionBbox_.min, LengthUnit{} );
        drawDimensionsVec3( "Box Max", selectionBbox_.max, LengthUnit{} );

        if ( selectionWorldBox_.valid() && bsizeStr != wbsizeStr )
            drawDimensionsVec3( "World Box Size", wbsize, LengthUnit{} );
    }

    if ( totalFaces || totalVerts || totalEdges || totalPoints )
    {
        ImGui::Spacing();
        ImGui::Spacing();

        drawPrimitivesInfo( "Triangles", totalFaces, totalSelectedFaces );
        drawPrimitivesInfo( "Vertices", totalVerts );
        drawPrimitivesInfo( "Edges", totalEdges, totalSelectedEdges );
        drawPrimitivesInfo( "Points", totalPoints, totalSelectedPoints );
    }

    if ( selectedObjs.size() == 1 && totalPoints )
        UI::inputTextCenteredReadOnly( "Point Normals", pointsHaveNormals ? "Yes" : "No", itemWidth, textColor, labelColor );

    if ( totalFaces )
    {
        drawUnitInfo( "Volume", totalVolume, VolumeUnit{} );

        ImGui::SetNextItemWidth( itemWidth );
        if ( totalSelectedArea > 0 )
        {
            UI::readOnlyValue<AreaUnit>( "Area", totalArea, selectedTextColor,
                { .decorationFormatString = valueToString<AreaUnit>( totalSelectedArea ) + " / {}" }, labelColor );
            UI::setTooltipIfHovered( "Selected / Total surface area", menu_scaling() );
        }
        else
        {
            UI::readOnlyValue<AreaUnit>( "Area", totalArea, textColor, {}, labelColor );
            UI::setTooltipIfHovered( "Total surface area", menu_scaling() );
        }
    }

    if ( totalLength > 0 )
        drawUnitInfo( "Length", totalLength, LengthUnit{} );

    if ( selectedObjs.size() == 1 && avgEdgeLen > 0 )
        drawUnitInfo( "Avg Edge Length", avgEdgeLen, LengthUnit{} );

    drawPrimitivesInfo( "Holes", holes );
    drawPrimitivesInfo( "Components", components );

#ifndef MRVIEWER_NO_VOXELS
    if ( selectedObjs.size() == 1 && selectedObjs.front()->asType<ObjectVoxels>() )
    {
        ImGui::Spacing();
        ImGui::Spacing();

        if ( isValidVoxelsInfo( voxelDims ) )
            drawDimensionsVec3( "Voxels Dims", *voxelDims, NoUnit{} );
        if ( isValidVoxelsInfo( voxelSize ) )
            drawDimensionsVec3( "Voxel Size", *voxelSize, LengthUnit{} );
        if ( isValidVoxelsInfo( voxelActiveBox ) )
        {
            if ( voxelDims && ( voxelActiveBox->min != Vector3i{} || voxelActiveBox->max != voxelDims ) )
            {
                drawDimensionsVec3( "Active Box Min", voxelActiveBox->min, NoUnit{} );
                drawDimensionsVec3( "Active Box Max", voxelActiveBox->max, NoUnit{} );
            }
        }
        if ( voxelMinValue && voxelIsoValue && voxelMaxValue )
        {
            drawDimensionsVec3( "Min,Iso,Max", Vector3f{ *voxelMinValue, *voxelIsoValue, *voxelMaxValue }, NoUnit{} );
        }
    }
#endif

    // Value for a dimension object.
    if ( selectedObjs.size() == 1 && selectedObjs.front()->asType<MeasurementObject>() )
    {
        ImGui::Spacing();
        ImGui::Spacing();

        auto* obj = selectedObjs.front().get();
        if ( auto* distance = obj->asType<DistanceMeasurementObject>() )
        {
            drawUnitInfo( "Distance", distance->computeDistance(), LengthUnit{} );
            const auto delta = distance->getWorldDelta();
            drawDimensionsVec3( "X/Y/Z Distance", Vector3f{ std::abs( delta.x ), std::abs( delta.y ), std::abs( delta.z ) }, LengthUnit{} );

            bool hasNominal = distance->hasComparisonReferenceValues();
            bool hasTolerance = distance->hasComparisonTolerances();

            ImVec2 buttonSize( ImGui::GetFrameHeight(), ImGui::GetFrameHeight() );

            { // Nominal distance.
                float nominalValue = 0;
                if ( hasNominal )
                    nominalValue = distance->getComparisonReferenceValue( 0 );

                ImGui::SetNextItemWidth( itemWidth );

                ImGui::PushStyleColor( ImGuiCol_FrameBg, originalFrameBgColor );
                MR_FINALLY{ ImGui::PopStyleColor(); };
                ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, originalFrameBorderSize );
                MR_FINALLY{ ImGui::PopStyleVar(); };

                if ( UI::input<LengthUnit>( "Nominal", nominalValue, -FLT_MAX, FLT_MAX, { .decorationFormatString = hasNominal ? "{}" : "Not specified" } ) )
                    distance->setComparisonReferenceValue( 0, nominalValue );

                if ( hasNominal )
                {
                    ImGui::SameLine();

                    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - buttonSize.x );
                    if ( UI::buttonEx( "\xC3\x97###removeNominal", buttonSize, { .customTexture = UI::getTexture( UI::TextureType::GradientBtnGray ).get() } ) ) // U+00D7 MULTIPLICATION SIGN
                    {
                        distance->resetComparisonReferenceValues();
                        distance->resetComparisonTolerances(); // This too.
                        hasNominal = false;
                        hasTolerance = false;
                    }

                    UI::setTooltipIfHovered( "Remove nominal value and tolerance", menu_scaling() );
                }
            }

            // Tolerance.
            if ( hasNominal ) // Sic!
            {
                ObjectComparableWithReference::ComparisonTolerance tol;
                if ( hasTolerance )
                    tol = distance->getComparisonTolerences( 0 );

                ImGui::SetNextItemWidth( itemWidthHalf1 );

                ImGui::PushStyleColor( ImGuiCol_FrameBg, originalFrameBgColor );
                MR_FINALLY{ ImGui::PopStyleColor(); };
                ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, originalFrameBorderSize );
                MR_FINALLY{ ImGui::PopStyleVar(); };

                if ( UI::input<LengthUnit>( "###positiveTolerance", tol.positive, 0.f, FLT_MAX, { .decorationFormatString = hasTolerance ? "{}" : "Not specified" } ) )
                    distance->setComparisonTolerance( 0, tol );

                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );

                ImGui::SetNextItemWidth( itemWidthHalf2 );

                if ( UI::input<LengthUnit>( "Tolerance##negative", tol.negative, -FLT_MAX, 0.f, { .decorationFormatString = hasTolerance ? "{}" : "Not specified" } ) )
                    distance->setComparisonTolerance( 0, tol );

                if ( hasTolerance )
                {
                    ImGui::SameLine();

                    ImGui::SetCursorPosX( ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - buttonSize.x );

                    if ( UI::buttonEx( "\xC3\x97###removeTolerance", buttonSize, { .customTexture = UI::getTexture( UI::TextureType::GradientBtnGray ).get() } ) ) // U+00D7 MULTIPLICATION SIGN
                        distance->resetComparisonTolerances();

                    UI::setTooltipIfHovered( "Remove tolerance", menu_scaling() );
                }
            }
        }
        else if ( auto* angle = obj->asType<AngleMeasurementObject>() )
            drawUnitInfo( "Angle", angle->computeAngle(), AngleUnit{} );
        else if ( auto* radius = obj->asType<RadiusMeasurementObject>() )
            drawUnitInfo( radius->getDrawAsDiameter() ? "Diameter" : "Radius", radius->computeRadiusOrDiameter(), LengthUnit{} );
    }

    drawCustomSelectionInformation_( selectedObjs, {
        .textColor = textColor,
        .labelColor = labelColor,
        .selectedTextColor = selectedTextColor,
        .itemWidth = itemWidth,
        .item2Width = getSceneInfoItemWidth_( 2 ),
        .item3Width = getSceneInfoItemWidth_( 3 ),
    } );

    // This looks a bit better.
    for ( int i = 0; i < 5; i++ )
        ImGui::Spacing();

    return resultingHeight();
}

void ImGuiMenu::drawFeaturePropertiesEditor_( const std::shared_ptr<Object>& object )
{
    auto& featureObject = dynamic_cast<FeatureObject&>( *object );

    const auto& list = featureObject.getAllSharedProperties();
    if ( !list.empty() )
        ImGui::Spacing();

    bool anyActive = false;

    std::size_t index = 0;

    for ( auto& prop : list )
    {
        std::visit( [&]( auto arg )
        {
            float speed = 0.01f;
            float min = std::numeric_limits<float>::lowest();
            float max = std::numeric_limits<float>::max();

            bool alreadyWasEditing = editedFeatureObject_.lock() == object;

            bool ret = false;

            if ( prop.kind == FeaturePropertyKind::position || prop.kind == FeaturePropertyKind::linearDimension )
            {
                ret = UI::drag<LengthUnit>( fmt::format( "{}##feature_property:{}", prop.propertyName, index ).c_str(), arg, speed, min, max );
            }
            else if ( prop.kind == FeaturePropertyKind::angle )
            {
                ret = UI::drag<AngleUnit>( fmt::format( "{}##feature_property:{}", prop.propertyName, index ).c_str(), arg, speed, min, max );
            }
            else
            {
                // `FeaturePropertyKind::direction` intentionally goes here.
                ret = UI::drag<NoUnit>( fmt::format( "{}##feature_property:{}", prop.propertyName, index ).c_str(), arg, speed, min, max );
            }

            if ( ret )
            {
                if ( !alreadyWasEditing  )
                {
                    editedFeatureObject_ = object;
                    editedFeatureObjectOldXf_ = object->xf();
                }

                prop.setter( arg, &featureObject, {} ); // Intentionally not using `viewer->viewport().id` here, setting globally is more intuitive.
            }

            if ( ImGui::IsItemDeactivatedAfterEdit() && editedFeatureObject_.lock() == object )
            {
                // Temporarily roll back the xf to write to the history.
                auto newXf = object->xf();
                object->setXf( editedFeatureObjectOldXf_ );
                AppendHistory<ChangeXfAction>( "Change Feature Transform", object );
                object->setXf( newXf );
            }

            if ( ImGui::IsItemActive() )
                anyActive = true;
        }, prop.getter( &featureObject, viewer->viewport().id ) );

        index++;
    }

    if ( !anyActive )
        editedFeatureObject_.reset();
}

bool ImGuiMenu::drawGeneralOptions( const std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    bool someChanges = false;
    const auto& selectedVisualObjs = SceneCache::getAllObjects<VisualObject, ObjectSelectivityType::Selected>();
    if ( !selectedVisualObjs.empty() )
    {
        const auto& viewportid = viewer->viewport().id;
        if ( make_visualize_checkbox( selectedVisualObjs, "Visibility", VisualizeMaskType::Visibility, viewportid ) )
        {
            someChanges = true;
            if ( sceneObjectsList_->getDeselectNewHiddenObjects() )
                for ( const auto& visObj : selectedVisualObjs )
                    if ( !visObj->isVisible( viewer->getPresentViewports() ) )
                        visObj->select( false );
        }
    }

    bool hasLocked = false, hasUnlocked = false;
    for ( const auto& s : selectedObjs )
    {
        if ( s->isLocked() )
            hasLocked = true;
        else
            hasUnlocked = true;
    }
    const bool mixedLocking = hasLocked && hasUnlocked;
    bool checked = hasLocked;
    someChanges |= UI::checkboxMixed( "Lock Transform", &checked, mixedLocking );
    if ( checked != hasLocked )
        for ( const auto& s : selectedObjs )
            s->setLocked( checked );

    return someChanges;
}

bool ImGuiMenu::drawAdvancedOptions( const std::vector<std::shared_ptr<VisualObject>>& selectedObjs, SelectedTypesMask selectedMask )
{
    if ( selectedObjs.empty() )
        return false;
    auto currWindow = ImGui::GetCurrentContext()->CurrentWindow;
    if ( currWindow )
        currWindow->DrawList->PushClipRect( currWindow->OuterRectClipped.Min, currWindow->OuterRectClipped.Max );
    if ( !RibbonButtonDrawer::CustomCollapsingHeader( "Advanced" ) )
    {
        if ( currWindow )
            currWindow->DrawList->PopClipRect();
        return false;
    }
    if ( currWindow )
        currWindow->DrawList->PopClipRect();

    const auto& viewportid = viewer->viewport().id;

    bool allIsObjMesh =
        selectedMask == SelectedTypesMask::ObjectMeshBit ||
        selectedMask == SelectedTypesMask::ObjectMeshHolderBit ||
        selectedMask == (SelectedTypesMask::ObjectMeshBit | SelectedTypesMask::ObjectMeshHolderBit);

    bool closePopup = false;

    if ( allIsObjMesh )
    {
        make_visualize_checkbox( selectedObjs, "Polygon Offset", MeshVisualizePropertyType::PolygonOffsetFromCamera, viewportid );
        make_width<ObjectMeshHolder>( selectedObjs, "Point size", [&] ( const ObjectMeshHolder* objMesh )
        {
            return objMesh->getPointSize();
        }, [&] ( ObjectMeshHolder* objMesh, float value )
        {
            objMesh->setPointSize( value );
        } );
    }

    make_light_strength( selectedObjs, "Shininess", [&] ( const VisualObject* obj )
    {
        return obj->getShininess();
    }, [&] ( VisualObject* obj, float value )
    {
        obj->setShininess( value );
    } );

    make_light_strength( selectedObjs, "Ambient Strength", [&] ( const VisualObject* obj )
    {
        return obj->getAmbientStrength();
    }, [&] ( VisualObject* obj, float value )
    {
        obj->setAmbientStrength( value );
    } );

    make_light_strength( selectedObjs, "Specular Strength", [&] ( const VisualObject* obj )
    {
        return obj->getSpecularStrength();
    }, [&] ( VisualObject* obj, float value )
    {
        obj->setSpecularStrength( value );
    } );

    bool allIsObjPoints = selectedMask == SelectedTypesMask::ObjectPointsHolderBit;

    if ( allIsObjPoints )
    {
        make_points_discretization( selectedObjs, "Visual Sampling", [&] ( const ObjectPointsHolder* data )
        {
            return data->getRenderDiscretization();
        }, [&] ( ObjectPointsHolder* data, const int val )
        {
            const auto AbsoluteMaxRenderingPoints = 1 << 25; // about 33M, rendering of more points will be slow anyway
            data->setMaxRenderingPoints( std::min( AbsoluteMaxRenderingPoints, val == 1 ?
                std::max( ObjectPointsHolder::MaxRenderingPointsDefault, int( data->numValidPoints() ) ) :
                int( data->numValidPoints() + val - 1 ) / val ) );
        } );
    }

    bool allIsFeatureObj = selectedMask == SelectedTypesMask::ObjectFeatureBit;
    if ( allIsFeatureObj )
    {
        const auto& selectedFeatureObjs = SceneCache::getAllObjects<FeatureObject, ObjectSelectivityType::Selected>();

        float minPointSize = 1, maxPointSize = 20;
        float minLineWidth = 1, maxLineWidth = 20;



        make_slider<float, FeatureObject>( selectedFeatureObjs, "Point size",
            [&] ( const FeatureObject* data ){ return data->getPointSize(); },
            [&]( FeatureObject* data, float value ){ data->setPointSize( value ); }, minPointSize, maxPointSize );
        make_slider<float, FeatureObject>( selectedFeatureObjs, "Line width",
            [&] ( const FeatureObject* data ){ return data->getLineWidth(); },
            [&]( FeatureObject* data, float value ){ data->setLineWidth( value ); }, minLineWidth, maxLineWidth );

        make_slider<float, FeatureObject>( selectedFeatureObjs, "Point subfeatures size",
            [&] ( const FeatureObject* data ){ return data->getSubfeaturePointSize(); },
            [&]( FeatureObject* data, float value ){ data->setSubfeaturePointSize( value ); }, minPointSize, maxPointSize );
        make_slider<float, FeatureObject>( selectedFeatureObjs, "Line subfeatures width",
            [&] ( const FeatureObject* data ){ return data->getSubfeatureLineWidth(); },
            [&]( FeatureObject* data, float value ){ data->setSubfeatureLineWidth( value ); }, minLineWidth, maxLineWidth );

        make_slider<float, FeatureObject>( selectedFeatureObjs, "Main component alpha",
            [&] ( const FeatureObject* data ){ return data->getMainFeatureAlpha(); },
            [&]( FeatureObject* data, float value ){ data->setMainFeatureAlpha( value ); }, 0, 1 );
        make_slider<float, FeatureObject>( selectedFeatureObjs, "Point subfeatures alpha",
            [&] ( const FeatureObject* data ){ return data->getSubfeatureAlphaPoints(); },
            [&]( FeatureObject* data, float value ){ data->setSubfeatureAlphaPoints( value ); }, 0, 1 );
        make_slider<float, FeatureObject>( selectedFeatureObjs, "Line subfeatures alpha",
            [&] ( const FeatureObject* data ){ return data->getSubfeatureAlphaLines(); },
            [&]( FeatureObject* data, float value ){ data->setSubfeatureAlphaLines( value ); }, 0, 1 );
        make_slider<float, FeatureObject>( selectedFeatureObjs, "Mesh subfeatures alpha",
            [&] ( const FeatureObject* data ){ return data->getSubfeatureAlphaMesh(); },
            [&]( FeatureObject* data, float value ){ data->setSubfeatureAlphaMesh( value ); }, 0, 1 );
    }

    return closePopup;
}

bool ImGuiMenu::drawRemoveButton( const std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    bool someChanges = false;
    auto backUpColorBtn = ImGui::GetStyle().Colors[ImGuiCol_Button];
    auto backUpColorBtnH = ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered];
    auto backUpColorBtnA = ImGui::GetStyle().Colors[ImGuiCol_ButtonActive];

    if ( !allowRemoval_ )
    {
        const auto& colorDis = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
        ImGui::GetStyle().Colors[ImGuiCol_Button] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = colorDis;
    }
    bool clicked = allowRemoval_ ?
        UI::button( "Remove", Vector2f( -1, 0 ) ) :
        ImGui::Button( "Remove", ImVec2( -1, 0 ) );
    if ( clicked )
    {
        someChanges |= true;
        if ( allowRemoval_ )
        {
            SCOPED_HISTORY( "Remove Objects (context)" );
            for ( int i = ( int )selectedObjs.size() - 1; i >= 0; --i )
                if ( selectedObjs[i] )
                {
                    // for now do it by one object
                    AppendHistory<ChangeSceneAction>( "Remove object", selectedObjs[i], ChangeSceneAction::Type::RemoveObject );
                    selectedObjs[i]->detachFromParent();
                }
        }
    }
    if ( !allowRemoval_ )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = backUpColorBtn;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = backUpColorBtnH;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = backUpColorBtnA;
    }

    return someChanges;
}

bool ImGuiMenu::drawDrawOptionsCheckboxes( const std::vector<std::shared_ptr<VisualObject>>& selectedVisualObjs, SelectedTypesMask selectedMask )
{
    bool someChanges = false;
    if ( selectedVisualObjs.empty() )
        return someChanges;

    bool allIsObjMesh =
        selectedMask == SelectedTypesMask::ObjectMeshBit ||
        selectedMask == SelectedTypesMask::ObjectMeshHolderBit ||
        selectedMask == ( SelectedTypesMask::ObjectMeshBit | SelectedTypesMask::ObjectMeshHolderBit );
    bool allIsObjLines = selectedMask == SelectedTypesMask::ObjectLinesHolderBit;
    bool allIsObjPoints = selectedMask == SelectedTypesMask::ObjectPointsHolderBit;
    bool allIsObjLabels = selectedMask == SelectedTypesMask::ObjectLabelBit;
    bool allIsFeatureObj = selectedMask == SelectedTypesMask::ObjectFeatureBit;

    const auto& viewportid = viewer->viewport().id;

    if ( allIsObjMesh )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Shading", MeshVisualizePropertyType::EnableShading, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Flat Shading", MeshVisualizePropertyType::FlatShading, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Edges", MeshVisualizePropertyType::Edges, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Points", MeshVisualizePropertyType::Points, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Selected Edges", MeshVisualizePropertyType::SelectedEdges, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Selected Tri-s", MeshVisualizePropertyType::SelectedFaces, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Borders", MeshVisualizePropertyType::BordersHighlight, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Triangles", MeshVisualizePropertyType::Faces, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Transparency", MeshVisualizePropertyType::OnlyOddFragments, viewportid );
        bool allHaveTexture = true;
        for ( const auto& visObj : selectedVisualObjs )
        {
            auto meshObj = visObj->asType<ObjectMeshHolder>();
            assert( meshObj );
            allHaveTexture = allHaveTexture &&
                ( !meshObj->getTexture().pixels.empty() && !meshObj->getUVCoords().empty() );
            if ( !allHaveTexture )
                break;
        }
        if ( allHaveTexture )
            someChanges |= make_visualize_checkbox( selectedVisualObjs, "Texture", MeshVisualizePropertyType::Texture, viewportid );
    }
    if ( allIsObjLines )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Points", LinesVisualizePropertyType::Points, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Smooth corners", LinesVisualizePropertyType::Smooth, viewportid );
        make_width<ObjectLinesHolder>( selectedVisualObjs, "Line width", [&] ( const ObjectLinesHolder* objLines )
        {
            return objLines->getLineWidth();
        }, [&] ( ObjectLinesHolder* objLines, float value )
        {
            objLines->setLineWidth( value );
        } );
        make_width<ObjectLinesHolder>( selectedVisualObjs, "Point size", [&] ( const ObjectLinesHolder* objLines )
        {
            return objLines->getPointSize();
        }, [&] ( ObjectLinesHolder* objLines, float value )
        {
            objLines->setPointSize( value );
        } );
    }
    if ( allIsObjPoints )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Selected Points", PointsVisualizePropertyType::SelectedVertices, viewportid );
        make_width<ObjectPointsHolder>( selectedVisualObjs, "Point size", [&] ( const ObjectPointsHolder* objPoints )
        {
            return objPoints->getPointSize();
        }, [&] ( ObjectPointsHolder* objPoints, float value )
        {
            objPoints->setPointSize( value );
        } );
    }
    if ( allIsObjLabels )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Always on top", VisualizeMaskType::DepthTest, viewportid, true );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Source point", LabelVisualizePropertyType::SourcePoint, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Background", LabelVisualizePropertyType::Background, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Contour", LabelVisualizePropertyType::Contour, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Leader line", LabelVisualizePropertyType::LeaderLine, viewportid );
    }
    if ( allIsFeatureObj )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Subfeatures", FeatureVisualizePropertyType::Subfeatures, viewportid );
    }
    someChanges |= make_visualize_checkbox( selectedVisualObjs, "Invert Normals", VisualizeMaskType::InvertedNormals, viewportid );
    someChanges |= make_visualize_checkbox( selectedVisualObjs, "Name", VisualizeMaskType::Name, viewportid );
    if ( allIsFeatureObj )
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Extra information next to name", FeatureVisualizePropertyType::DetailsOnNameTag, viewportid );
    if ( viewer->experimentalFeatures )
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Clipping", VisualizeMaskType::ClippedByPlane, viewportid );

    { // Dimensions checkboxes.
        bool fail = false;

        // Which dimensions our objects have.
        // All objects must have the same values here, otherwise we don't draw anything.
        bool supportedDimensions[std::size_t( DimensionsVisualizePropertyType::_count )]{};
        bool firstObject = true;

        for ( const auto& object : selectedVisualObjs )
        {
            for ( std::size_t i = 0; i < std::size_t( DimensionsVisualizePropertyType::_count ); i++ )
            {
                bool value = object->supportsVisualizeProperty( DimensionsVisualizePropertyType( i ) );
                if ( firstObject )
                    supportedDimensions[i] = value;
                else if ( supportedDimensions[i] != value )
                {
                    fail = true;
                    break;
                }
            }
            firstObject = false;
        }

        if ( !fail && !firstObject )
        {
            for ( std::size_t i = 0; i < std::size_t( DimensionsVisualizePropertyType::_count ); i++ )
            {
                if ( !supportedDimensions[i] )
                    continue;

                auto enumValue = DimensionsVisualizePropertyType( i );

                someChanges |= make_visualize_checkbox( selectedVisualObjs, toString( enumValue ).data(), enumValue, viewportid );
            }
        }
    }

    return someChanges;
}

bool ImGuiMenu::drawDrawOptionsColors( const std::vector<std::shared_ptr<VisualObject>>& selectedVisualObjs )
{
    bool someChanges = false;
    const auto& selectedMeshObjs = SceneCache::getAllObjects<ObjectMeshHolder, ObjectSelectivityType::Selected>();
    const auto& selectedPointsObjs = SceneCache::getAllObjects<ObjectPointsHolder, ObjectSelectivityType::Selected>();
    const auto& selectedLabelObjs = SceneCache::getAllObjects<ObjectLabel, ObjectSelectivityType::Selected>();
    const auto& selectedFeatureObjs = SceneCache::getAllObjects<FeatureObject, ObjectSelectivityType::Selected>();
    if ( selectedVisualObjs.empty() )
        return someChanges;

    if ( getViewerInstance().viewport_list.size() > 1 )
    {
        ImGui::SetNextItemWidth( 75.0f * menu_scaling() );

        if (ImGui::BeginCombo( "Viewport Id",
            selectedViewport_.value() == 0 ? "Default" :
            std::to_string( selectedViewport_.value() ).c_str() ) )
        {
            if ( ImGui::Selectable( "Default" ) )
                selectedViewport_ = ViewportId{ 0 };

            for ( const auto& viewport : getViewerInstance().viewport_list )
            {
                if ( ImGui::Selectable( std::to_string( viewport.id.value() ).c_str() ) )
                    selectedViewport_ = viewport.id;
            }

            ImGui::EndCombo();
        }
    }

    make_color_selector<VisualObject>( selectedVisualObjs, ("Selected color##" + std::to_string(selectedViewport_.value())).c_str(), [&] ( const VisualObject* data )
    {
        return Vector4f( data->getFrontColor(true, selectedViewport_ ) );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setFrontColor( Color( color ), true, selectedViewport_ );
    } );
    make_color_selector<VisualObject>( selectedVisualObjs, "Unselected color", [&] ( const VisualObject* data )
    {
        return Vector4f( data->getFrontColor( false, selectedViewport_ ) );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setFrontColor( Color( color ), false, selectedViewport_ );
    } );
    make_color_selector<VisualObject>( selectedVisualObjs, "Back Triangles color", [&] ( const VisualObject* data )
    {
        return Vector4f( data->getBackColor( selectedViewport_ ) );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setBackColor( Color( color ), selectedViewport_ );
    } );

    if ( !selectedMeshObjs.empty() )
    {
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Edges color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getEdgesColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setEdgesColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Points color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getPointsColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setPointsColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Selected Tri-s color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getSelectedFacesColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setSelectedFacesColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Selected Edges color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getSelectedEdgesColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setSelectedEdgesColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Borders color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getBordersColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setBordersColor( Color( color ), selectedViewport_ );
        } );
    }
    if ( !selectedPointsObjs.empty() )
    {
        make_color_selector<ObjectPointsHolder>( selectedPointsObjs, "Selected Points color", [&] ( const ObjectPointsHolder* data )
        {
            return Vector4f( data->getSelectedVerticesColor( selectedViewport_ ) );
        }, [&] ( ObjectPointsHolder* data, const Vector4f& color )
        {
            data->setSelectedVerticesColor( Color( color ), selectedViewport_ );
        } );
    }
    if ( !selectedLabelObjs.empty() )
    {
        make_color_selector<ObjectLabel>( selectedLabelObjs, "Source point color", [&] ( const ObjectLabel* data )
        {
            return Vector4f( data->getSourcePointColor( selectedViewport_ ) );
        }, [&] ( ObjectLabel* data, const Vector4f& color )
        {
            data->setSourcePointColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectLabel>( selectedLabelObjs, "Leader line color", [&] ( const ObjectLabel* data )
        {
            return Vector4f( data->getLeaderLineColor( selectedViewport_ ) );
        }, [&] ( ObjectLabel* data, const Vector4f& color )
        {
            data->setLeaderLineColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectLabel>( selectedLabelObjs, "Contour color", [&] ( const ObjectLabel* data )
        {
            return Vector4f( data->getContourColor( selectedViewport_ ) );
        }, [&] ( ObjectLabel* data, const Vector4f& color )
        {
            data->setContourColor( Color( color ), selectedViewport_ );
        } );
    }

    if ( !selectedFeatureObjs.empty() )
    {
        make_color_selector<FeatureObject>( selectedFeatureObjs, "Decorations color (selected)", [&] ( const FeatureObject* data )
        {
            return Vector4f( data->getDecorationsColor( true, selectedViewport_ ) );
        }, [&] ( FeatureObject* data, const Vector4f& color )
        {
            data->setDecorationsColor( Color( color ), true, selectedViewport_ );
        } );

        make_color_selector<FeatureObject>( selectedFeatureObjs, "Decorations color (unselected)", [&] ( const FeatureObject* data )
        {
            return Vector4f( data->getDecorationsColor( false, selectedViewport_ ) );
        }, [&] ( FeatureObject* data, const Vector4f& color )
        {
            data->setDecorationsColor( Color( color ), false, selectedViewport_ );
        } );
    }

    if ( !selectedVisualObjs.empty() )
    {
        make_slider<std::uint8_t, VisualObject>( selectedVisualObjs, "Opacity", [&] ( const VisualObject* data )
        {
            return data->getGlobalAlpha( selectedViewport_ );
        }, [&] ( VisualObject* data, uint8_t alpha )
        {
            data->setGlobalAlpha( alpha, selectedViewport_ );
        }, 0, 255 );
    }

    return someChanges;
}

void ImGuiMenu::drawCustomSelectionInformation_( const std::vector<std::shared_ptr<Object>>&, const SelectionInformationStyle& )
{
}

void ImGuiMenu::draw_custom_selection_properties( const std::vector<std::shared_ptr<Object>>& )
{}

void ImGuiMenu::drawTagInformation_( const std::vector<std::shared_ptr<Object>>& selected, const SelectionInformationStyle& style )
{
    const auto initWidth = ImGui::GetContentRegionAvail().x;
    const auto initCursorScreenPos = ImGui::GetCursorScreenPos();
    const auto initCursorPos = ImGui::GetCursorPos();
    const auto itemInnerSpacing = ImGui::GetStyle().ItemInnerSpacing;
    const auto textLineHeight = ImGui::GetTextLineHeight();

    if ( ImGui::InvisibleButton( "##EnterTagsWindow", { style.itemWidth, textLineHeight } ) )
        ImGui::OpenPopup( "TagsPopup" );

    static const auto setIntersect = [] <typename T> ( const std::set<T>& a, const std::set<T>& b )
    {
        std::set<T> result;
        std::set_intersection( a.begin(), a.end(), b.begin(), b.end(), std::inserter( result, result.begin() ) );
        return result;
    };
    static const auto setUnion = [] <typename T> ( const std::set<T>& a, const std::set<T>& b )
    {
        std::set<T> result;
        std::set_union( a.begin(), a.end(), b.begin(), b.end(), std::inserter( result, result.begin() ) );
        return result;
    };

    assert( !selected.empty() );
    auto allTags = selected.front()->tags();
    auto commonTags = allTags;
    for ( auto i = 1; i < selected.size(); ++i )
    {
        const auto& selObj = selected[i];
        allTags = setUnion( allTags, selObj->tags() );
        commonTags = setIntersect( commonTags, selObj->tags() );
    }

    static const auto hiddenTagPred = [] ( const std::string& tag )
    {
        // hide service tags starting with a dot
        return tag.starts_with( '.' );
    };
    std::erase_if( allTags, hiddenTagPred );
    std::erase_if( commonTags, hiddenTagPred );

    std::ostringstream oss;
    size_t tagCount = 0;
    for ( const auto& tag : commonTags )
    {
        if ( tagCount++ != 0 )
            oss << ", ";
        oss << tag;
    }
    auto text = oss.str();
    if ( const auto uncommonTagCount = allTags.size() - commonTags.size() )
        text += ( tagCount != 0 ? " + " : "" ) + fmt::format( "{} uncommon tag{}", uncommonTagCount, uncommonTagCount != 1 ? "s" : "" );
    if ( text.empty() )
        text = "–";

    auto textSize = ImGui::CalcTextSize( text.c_str() );
    if ( style.itemWidth < textSize.x )
    {
        // TODO: cache
        const auto ellipsisSize = ImGui::CalcTextSize( "..." );
        auto textLen = text.size();
        for ( --textLen; textLen > 0; --textLen )
        {
            textSize = ImGui::CalcTextSize( text.data(), text.data() + textLen );
            if ( textSize.x + ellipsisSize.x <= style.itemWidth )
                break;
        }
        text = text.substr( 0, textLen ) + "...";
        textSize = ImGui::CalcTextSize( text.c_str() );
    }

    const auto offset = std::floor( ( style.itemWidth - textSize.x ) * 0.5f );
    ImGui::SetCursorPos( { initCursorPos.x + offset, initCursorPos.y } );
    ImGui::TextColored( style.textColor, "%s", text.c_str() );

    ImGui::SetCursorPos( { initCursorPos.x + style.itemWidth + itemInnerSpacing.x, initCursorPos.y } );
    ImGui::TextColored( style.labelColor, "Tags" );

    static const auto BeginPopup2 = [] ( const char* name, ImVec2 size, const ImVec2* pos = nullptr )
    {
        // https://github.com/ocornut/imgui/issues/6443#issuecomment-1556039133
        auto& g = *GImGui;
        if ( g.OpenPopupStack.Size <= g.BeginPopupStack.Size )
        {
            g.NextWindowData.ClearFlags();
            return false;
        }

        auto* window = ImGui::FindWindowByName( name );
        const auto [initialWindowPos, haveSavedWindowPos] = ImGui::LoadSavedWindowPos( name, window, size.y, pos );
        UI::getDefaultWindowRectAllocator().setFreeNextWindowPos( name, initialWindowPos, haveSavedWindowPos ? ImGuiCond_FirstUseEver : ImGuiCond_Appearing, ImVec2( 0, 0 ) );
        MR_FINALLY {
            ImGui::SaveWindowPosition( name, window );
        };

        ImGui::SetNextWindowSize( size, ImGuiCond_Appearing );
        return ImGui::BeginPopupEx( g.CurrentWindow->GetID( name ), ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings );
    };

    ImGui::PushStyleColor( ImGuiCol_PopupBg, ImGui::GetStyleColorVec4( ImGuiCol_WindowBg ) );
    if ( BeginPopup2( "TagsPopup", { initWidth, -1 }, &initCursorScreenPos ) )
    {
        if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
            ImGui::CloseCurrentPopup();

        auto* iconsFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
        if ( iconsFont )
            iconsFont->Scale = cDefaultFontSize / cBigIconSize;

        const auto buttonWidth = [&] ( const char* label )
        {
            return ImGui::GetStyle().FramePadding.x * 2.f + ImGui::CalcTextSize( label, NULL, true ).x;
        };
        if ( iconsFont )
            ImGui::PushFont( iconsFont );
        const auto* removeButtonText = iconsFont ? "\xef\x80\x8d" : "X";
        const auto* addButtonText = iconsFont ? "\xef\x81\x95" : "+";
        const auto removeButtonWidth = buttonWidth( removeButtonText );
        const auto addButtonWidth = buttonWidth( addButtonText );
        if ( iconsFont )
            ImGui::PopFont();

        const auto& allVisTags = VisualObjectTagManager::tags();
        auto allKnownTags = allTags;
        for ( const auto& [tag, _] : allVisTags )
            allKnownTags.emplace( tag );

        for ( const auto& tag : commonTags )
        {
            const auto tagButtonWidth = buttonWidth( tag.c_str() ) + removeButtonWidth;
            if ( ImGui::GetContentRegionAvail().x < tagButtonWidth )
                ImGui::NewLine();

            const auto initCursorPosX = ImGui::GetCursorPosX();

            if ( allVisTags.contains( tag ) )
            {
                const auto color = allVisTags.at( tag ).selectedColor;
                ImGui::PushStyleColor( ImGuiCol_Button, color );
            }

            ImGui::PushStyleVar( ImGuiStyleVar_ButtonTextAlign, { 0.0f, 0.5f } );
            ImGui::SetNextItemAllowOverlap();
            if ( ImGui::Button( tag.c_str(), { tagButtonWidth, 0 } ) )
            {
                std::optional<VisualObjectTag> visTag;
                if ( auto it = allVisTags.find( tag ); it != allVisTags.end() )
                    visTag = it->second;

                tagEditorState_ = {
                    .initName = tag,
                    .name = tag,
                    .initHasFrontColor = bool( visTag ),
                    .hasFrontColor = bool( visTag ),
                };
                if ( visTag )
                {
                    tagEditorState_.selectedColor = visTag->selectedColor;
                    tagEditorState_.unselectedColor = visTag->unselectedColor;
                }
                else
                {
                    tagEditorState_.selectedColor = SceneColors::get( SceneColors::SelectedObjectMesh );
                    tagEditorState_.unselectedColor = SceneColors::get( SceneColors::UnselectedObjectMesh );
                }

                showEditTag_ = true;
            }
            ImGui::PopStyleVar();

            if ( allVisTags.contains( tag ) )
                ImGui::PopStyleColor();

            ImGui::SameLine( initCursorPosX + buttonWidth( tag.c_str() ), 0 );
            if ( iconsFont )
                ImGui::PushFont( iconsFont );
            ImGui::PushStyleColor( ImGuiCol_Button, Color{ 0xff, 0xff, 0xff, 0x00 } );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, Color{ 0xff, 0x5f, 0x5f } );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, Color::red() );
            const auto closeButtonLabel = removeButtonText + fmt::format( "##{}", tag );
            if ( ImGui::Button( closeButtonLabel.c_str() ) )
            {
                for ( const auto& selObj : selected )
                    selObj->removeTag( tag );
            }
            ImGui::PopStyleColor( 3 );
            if ( iconsFont )
                ImGui::PopFont();

            ImGui::SameLine();
        }

        if ( tagCount != 0 )
        {
            ImGui::NewLine();
            ImGui::Spacing();
        }

        if ( ImGui::IsWindowAppearing() )
        {
            ImGui::SetKeyboardFocusHere();
            tagNewName_.clear();
        }

        // completion callback for ImGui
        // called every time user presses the Tab key
        // completes the existing tag name if its prefix is typed
        static const auto tagCompletion = [] ( ImGuiInputTextCallbackData* data ) -> int
        {
            if ( data->EventFlag == ImGuiInputTextFlags_CallbackCompletion )
            {
                std::string_view text{ data->Buf, (size_t)data->BufTextLen };
                const auto& allKnownTags = *(std::set<std::string>*)data->UserData;
                std::string_view candidate;
                for ( const auto& tag : allKnownTags )
                {
                    if ( tag.starts_with( text ) )
                    {
                        if ( candidate.empty() )
                        {
                            candidate = tag;
                        }
                        else
                        {
                            candidate = {};
                            break;
                        }
                    }
                }
                if ( !candidate.empty() )
                {
                    data->InsertChars( data->CursorPos, candidate.substr( text.length() ).data() );
                    data->ClearSelection();
                    data->SelectionStart = (int)text.length();
                }
            }
            return 0;
        };
        ImGui::SetNextItemWidth( ImGui::GetContentRegionAvail().x - itemInnerSpacing.x - addButtonWidth );
        if ( ImGui::InputTextWithHint( "##TagNew", "Type to add new tag...", &tagNewName_, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion, tagCompletion, &allKnownTags ) )
        {
            for ( const auto& selObj : selected )
                selObj->addTag( tagNewName_ );
            tagNewName_.clear();
        }

        if ( iconsFont )
            ImGui::PushFont( iconsFont );
        ImGui::SameLine( 0, itemInnerSpacing.x );
        if ( ImGui::Button( addButtonText ) )
        {
            for ( const auto& selObj : selected )
                selObj->addTag( tagNewName_ );
            tagNewName_.clear();
        }
        if ( iconsFont )
            ImGui::PopFont();

        ImGui::EndPopup();
    }
    ImGui::PopStyleColor();
}

float ImGuiMenu::drawTransform_()
{
    const auto& selected = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();

    const auto scaling = menu_scaling();
    auto& style = ImGui::GetStyle();

    float resultHeight_ = 0.f;
    if ( selected.size() == 1 && !selected[0]->isLocked() )
    {
        if ( !selectionChangedToSingleObj_ )
        {
            selectionChangedToSingleObj_ = true;
            sceneObjectsList_->setNextFrameFixScroll();
        }
        resultHeight_ = ImGui::GetTextLineHeight() + style.FramePadding.y * 2 + style.ItemSpacing.y;
        bool openedContext = false;
        if ( drawCollapsingHeaderTransform_() )
        {
            openedContext = drawTransformContextMenu_( selected[0] );
            const float transformHeight = ( ImGui::GetTextLineHeight() + style.FramePadding.y * 2 ) * 3 + style.ItemSpacing.y * 2;
            resultHeight_ += transformHeight + style.ItemSpacing.y;
            ImGui::BeginChild( "SceneTransform", ImVec2( 0, transformHeight ) );
            auto& data = *selected.front();

            auto xf = data.xf();
            Matrix3f q, r;
            decomposeMatrix3( xf.A, q, r );

            auto euler = ( 180 / PI_F ) * q.toEulerAngles();
            Vector3f scale{ r.x.x, r.y.y, r.z.z };

            bool inputDeactivated = false;
            bool inputChanged = false;

            ImGui::PushItemWidth( getSceneInfoItemWidth_( 3 ) );
            if ( uniformScale_ )
            {
                float midScale = ( scale.x + scale.y + scale.z ) / 3.0f;
                ImGui::SetNextItemWidth( getSceneInfoItemWidth_() );
                inputChanged = UI::drag<NoUnit>( "##scaleX", midScale, midScale * 0.01f, 1e-3f, 1e+6f );
                if ( inputChanged )
                    scale.x = scale.y = scale.z = midScale;
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine();
            }
            else
            {
                inputChanged = UI::drag<NoUnit>( "##scaleX", scale.x, scale.x * 0.01f, 1e-3f, 1e+6f );
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
                inputChanged = UI::drag<NoUnit>( "##scaleY", scale.y, scale.y * 0.01f, 1e-3f, 1e+6f ) || inputChanged;
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
                inputChanged = UI::drag<NoUnit>( "##scaleZ", scale.z, scale.z * 0.01f, 1e-3f, 1e+6f ) || inputChanged;
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
            }

            auto ctx = ImGui::GetCurrentContext();
            assert( ctx );
            auto window = ctx->CurrentWindow;
            assert( window );
            auto diff = ImGui::GetStyle().FramePadding.y - cCheckboxPadding * menu_scaling();
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + diff );
            UI::checkbox( "Uni-scale", &uniformScale_ );
            window->DC.CursorPosPrevLine.y -= diff;
            UI::setTooltipIfHovered( "Selects between uniform scaling or separate scaling along each axis", scaling );
            ImGui::PopItemWidth();

            ImGui::SetNextItemWidth( getSceneInfoItemWidth_() );
            bool rotationChanged = UI::drag<AngleUnit>( "Rotation XYZ", euler, invertedRotation_ ? -0.1f : 0.1f, -360.f, 360.f, { .sourceUnit = AngleUnit::degrees } );
            bool rotationDeactivatedAfterEdit = ImGui::IsItemDeactivatedAfterEdit();
            ImGui::SetItemTooltip( "%s", "Rotation round [X, Y, Z] axes respectively." );
            inputChanged = inputChanged || rotationChanged;
            inputDeactivated = inputDeactivated || rotationDeactivatedAfterEdit;
            if ( ImGui::IsItemHovered() )
            {
                ImGui::BeginTooltip();
                ImGui::Text( "Sequential intrinsic rotations around Oz, Oy and Ox axes." ); // see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations
                ImGui::EndTooltip();
            }

            constexpr float cZenithEps = 0.01f;
            if ( rotationChanged && ImGui::IsMouseDragging( ImGuiMouseButton_Left ) )
            {
                // resolve singularity
                if ( std::fabs( euler.y ) > 90.f - cZenithEps )
                {
                    euler.x = euler.x > 0.f ? euler.x - 180.f : euler.x + 180.f;
                    euler.z = euler.z > 0.f ? euler.z - 180.f : euler.z + 180.f;
                    invertedRotation_ = !invertedRotation_;
                    euler.y = euler.y > 0.f ? 90.f - cZenithEps : -90.f + cZenithEps;
                }
            }
            if ( rotationDeactivatedAfterEdit )
            {
                invertedRotation_ = false;
            }
            euler.y = std::clamp( euler.y, -90.0f + 2.0f * cZenithEps, 90.0f - 2.0f * cZenithEps );
            if ( inputChanged )
                xf.A = Matrix3f::rotationFromEuler( ( PI_F / 180 ) * euler ) * Matrix3f::scale( scale );

            const auto trSpeed = ( selectionBbox_.valid() && selectionBbox_.diagonal() > std::numeric_limits<float>::epsilon() ) ? 0.003f * selectionBbox_.diagonal() : 0.003f;

            ImGui::SetNextItemWidth( getSceneInfoItemWidth_() );
            auto wbsize = selectionWorldBox_.valid() ? selectionWorldBox_.size() : Vector3f::diagonal( 1.f );
            auto minSizeDim = wbsize.length();
            if ( minSizeDim == 0 )
                minSizeDim = 1.f;
            auto translation = xf.b;
            auto translationChanged = UI::drag<LengthUnit>( "Translation", translation, trSpeed, -cMaxTranslationMultiplier * minSizeDim, +cMaxTranslationMultiplier * minSizeDim );
            inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();

            if ( translationChanged )
                xf.b = translation;

            if ( xfHistUpdated_ )
                xfHistUpdated_ = !inputDeactivated;

            if ( xf != data.xf() && !xfHistUpdated_ )
            {
                AppendHistory<ChangeXfAction>( "Manual Change Transform", selected[0] );
                xfHistUpdated_ = true;
            }
            data.setXf( xf );
            ImGui::EndChild();
            if ( !openedContext )
                openedContext = drawTransformContextMenu_( selected[0] );
        }
        if ( !openedContext )
            drawTransformContextMenu_( selected[0] );
    }
    else
    {
        if ( selectionChangedToSingleObj_ )
            selectionChangedToSingleObj_ = false;
    }

    return resultHeight_;
}

bool ImGuiMenu::drawCollapsingHeader_( const char* label, ImGuiTreeNodeFlags flags )
{
    return ImGui::CollapsingHeader( label, flags );
}

bool ImGuiMenu::drawCollapsingHeaderTransform_()
{
    return drawCollapsingHeader_( "Transform", ImGuiTreeNodeFlags_DefaultOpen );
}

bool ImGuiMenu::make_visualize_checkbox( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label, AnyVisualizeMaskEnum type, MR::ViewportMask viewportid, bool invert /*= false*/ )
{
    auto realRes = getRealValue( selectedVisualObjs, type, viewportid, invert );
    bool checked = realRes.first;
    const bool res = UI::checkboxMixed( label, &checked, !realRes.second && realRes.first );
    if ( checked != realRes.first )
    {
        if ( invert )
            checked = !checked;
        for ( const auto& data : selectedVisualObjs )
            if ( data )
                data->setVisualizeProperty( checked, type, viewportid );
    }

    return res;
}

template<typename ObjectT>
void ImGuiMenu::make_color_selector( std::vector<std::shared_ptr<ObjectT>> selectedVisualObjs, const char* label,
                                std::function<Vector4f( const ObjectT* )> getter,
                                std::function<void( ObjectT*, const Vector4f& )> setter )
{
    auto color = getter( selectedVisualObjs[0].get() );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i].get() ) != color )
        {
            isAllTheSame = false;
            break;
        }
    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( !isAllTheSame )
    {
        color = Vector4f::diagonal( 0.0f ); color[3] = 1.0f;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }

    std::string storedName = label;
    for ( const auto& obj : selectedVisualObjs )
        storedName += std::to_string( intptr_t( obj.get() ) );

    const auto colorConstForComparation = color;
    color = getStoredColor_( storedName, Color( color ) );
    ImGui::PushItemWidth( 40 * menu_scaling() );
    if ( ImGui::ColorEdit4( label, &color.x,
        ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel ) )
        storedColor_ = { storedName,color };
    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( color != colorConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data.get(), color );
}

void ImGuiMenu::make_light_strength( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label,
    std::function<float( const VisualObject* )> getter,
    std::function<void( VisualObject*, const float& )> setter
)
{
    if ( selectedVisualObjs.empty() )
        return;

    auto obj = selectedVisualObjs[0];
    auto value = getter( obj.get() );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i].get() ) != value )
        {
            isAllTheSame = false;
            break;
        }

    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( !isAllTheSame )
    {
        value = 0.f;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }
    const auto valueConstForComparation = value;

    ImGui::PushItemWidth( 50 * menu_scaling() );
    UI::drag<NoUnit>( label, value, 0.01f, -99.0f, 99.0f );

    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( value != valueConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data.get(), value );
}

template <typename T, typename ObjectType>
void ImGuiMenu::make_slider( std::vector<std::shared_ptr<ObjectType>> selectedVisualObjs, const char* label,
    std::function<T( const ObjectType* )> getter, std::function<void( ObjectType*, T )> setter, T min, T max )
{
    if ( selectedVisualObjs.empty() )
        return;

    auto obj = selectedVisualObjs[0];
    auto value = getter( obj.get() );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i].get() ) != value )
        {
            isAllTheSame = false;
            break;
        }

    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( !isAllTheSame )
    {
        value = max;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }
    const auto valueConstForComparation = value;

    ImGui::PushItemWidth( 100 * menu_scaling() );
    UI::slider<NoUnit>( label, value, min, max );

    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( value != valueConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data.get(), T( value ) );
}

template<typename ObjType>
void ImGuiMenu::make_width( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label,
    std::function<float( const ObjType* )> getter,
    std::function<void( ObjType*, const float& )> setter )
{
    auto objLines = selectedVisualObjs[0]->asType<ObjType>();
    auto value = getter( objLines );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i]->asType<ObjType>() ) != value )
        {
            isAllTheSame = false;
            break;
        }
    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( !isAllTheSame )
    {
        value = 0.f;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }
    const auto valueConstForComparation = value;

    ImGui::PushItemWidth( 50 * menu_scaling() );
    UI::drag<PixelSizeUnit>( label, value, 0.02f, 0.5f, 30.0f );
    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( value != valueConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data->asType<ObjType>(), value );
}

void ImGuiMenu::make_points_discretization( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label,
    std::function<int( const ObjectPointsHolder* )> getter,
    std::function<void( ObjectPointsHolder*, const int& )> setter )
{
    auto objPoints = selectedVisualObjs[0]->asType<ObjectPointsHolder>();
    auto value = getter( objPoints );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i]->asType<ObjectPointsHolder>() ) != value )
        {
            isAllTheSame = false;
            break;
        }

    if ( !isAllTheSame )
    {
        value = 1;
    }
    const auto valueConstForComparation = value;

    ImGui::SetNextItemWidth( 50 * menu_scaling() );
    UI::drag<NoUnit>( label, value, 0.1f, 1, 9999, {}, UI::defaultSliderFlags, 0, 0 );

    if ( value != valueConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data->asType<ObjectPointsHolder>(), value);
}

Vector4f ImGuiMenu::getStoredColor_( const std::string& str, const Color& defaultColor ) const
{
    if ( !storedColor_ || storedColor_->first != str )
        return Vector4f( defaultColor );
    return storedColor_->second;
}

void ImGuiMenu::draw_custom_plugins()
{
    pluginsCache_.validate( viewer->plugins );
    StateBasePlugin* enabled = pluginsCache_.findEnabled();

    float availibleWidth = 200.0f * menu_scaling();

    const auto& selectedObjects = SceneCache::getAllObjects<const Object, ObjectSelectivityType::Selected>();
    const auto& selectedVisObjects = SceneCache::getAllObjects<VisualObject, ObjectSelectivityType::Selected>();

    ImGui::SetNextWindowPos( ImVec2( 410.0f * menu_scaling(), 0 ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowSize( ImVec2( 0.0f, 0.0f ), ImGuiCond_FirstUseEver );
    ImGui::Begin( "Plugins", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize );

    ImGui::SetCursorPosX( 570.f );
    if ( UI::inputText( "Search", searchPluginsString_ ) )
    {
        Viewer::instanceRef().incrementForceRedrawFrames( 2, true );
    }

    auto& colors = ImGui::GetStyle().Colors;
    auto backUpButtonColor = colors[ImGuiCol_Button];
    auto backUpTextColor = colors[ImGuiCol_Text];

    ImGui::BeginTabBar( "##CustomPluginsTabs", ImGuiTabBarFlags_TabListPopupButton );

    const int pluginsPerLine = 4;
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        StatePluginTabs tab = StatePluginTabs( t );
        const auto& plugins = pluginsCache_.getTabPlugins( tab );
        int counter = 0;
        for ( auto& plugin : plugins )
        {
            if ( !plugin->checkStringMask( searchPluginsString_ ) )
                continue;

            if ( counter == 0 )
            {
                if ( !ImGui::BeginTabItem( StateBasePlugin::getTabName( tab ) ) )
                    break;
            }

            std::string requirements = plugin->isAvailable( selectedObjects );
            bool canEnable = !enabled && requirements.empty();

            if ( plugin->isEnabled() )
            {
                colors[ImGuiCol_Button] = { 0.0f,0.8f,0.0f,1.0f };
                colors[ImGuiCol_Text] = { 0.0f,0.0f,0.0f,1.0f };
            }
            else if ( !canEnable )
            {
                colors[ImGuiCol_Button] = { 0.5f,0.5f,0.5f,1.0f };
                colors[ImGuiCol_Text] = { 1.0f,1.0f,1.0f,1.0f };
            }

            if ( counter % pluginsPerLine != 0 )
                ImGui::SameLine();

            if ( ImGui::Button( plugin->uiName().c_str(), ImVec2(availibleWidth, 0)) )
            {
                if ( plugin->isEnabled() )
                    plugin->enable( false );
                else if ( canEnable )
                    plugin->enable( true );
            }
            ++counter;

            colors[ImGuiCol_Text] = { 1.0f,1.0f,1.0f,1.0f };
            const auto strTooltip = plugin->getTooltip();
            if ( ImGui::IsItemHovered() && ( !strTooltip.empty() || !requirements.empty() ) )
            {
                ImVec2 textSize;
                if ( requirements.empty() )
                    textSize = ImGui::CalcTextSize( strTooltip.c_str(), NULL, false, 400.f );
                else
                {
                    if ( strTooltip.empty() )
                        textSize = ImGui::CalcTextSize( requirements.c_str(), NULL, false, 400.f );
                    else
                        textSize = ImGui::CalcTextSize( ( strTooltip + "\n" + requirements ).c_str(), NULL, false, 400.f );
                }
                ImGui::SetNextWindowContentSize( textSize );
                ImGui::BeginTooltip();
                if ( !strTooltip.empty() )
                {
                    ImGui::TextWrapped( "%s", strTooltip.c_str() );
                }
                if ( !requirements.empty() )
                {
                    ImGui::PushStyleColor( ImGuiCol_Text, Color::red().getUInt32() );
                    ImGui::TextWrapped( "%s", requirements.c_str() );
                    ImGui::PopStyleColor();
                }
                ImGui::EndTooltip();
            }

            colors[ImGuiCol_Button] = backUpButtonColor;
            colors[ImGuiCol_Text] = backUpTextColor;
        }

        int counterModifier = 0;
        if ( !selectedVisObjects.empty() )
        {
            if ( counter != 0 )
                ImGui::Separator();

            for ( const auto& modifier : modifiers_ )
            {
                if ( tab != modifier->getTab() )
                    continue;

                if ( !modifier->checkStringMask( searchPluginsString_ ) )
                    continue;

                if ( counter + counterModifier == 0 )
                {
                    if ( !ImGui::BeginTabItem( StateBasePlugin::getTabName( tab ) ) )
                        break;
                }

                if ( counterModifier % pluginsPerLine != 0 )
                    ImGui::SameLine();

                if ( ImGui::Button( modifier->name().c_str(), ImVec2( availibleWidth, 0 ) ) )
                {
                    // Here should be popups
                    if ( modifier->modify( selectedVisObjects ) )
                        spdlog::info( "{}: success", modifier->name() );
                    else
                        spdlog::warn( "{}: failure", modifier->name() );
                }
                ++counterModifier;
            }
        }
        if ( ( counter + counterModifier ) != 0 )
        {
            ImGui::EndTabItem();
        }
    }

    ImGui::SetWindowSize( ImGui::GetWindowSize() );
    ImGui::EndTabBar();
    ImGui::End();

    if ( enabled && enabled->isEnabled() )
    {
        if ( allowRemoval_ )
            allowRemoval_ = false;
        enabled->drawDialog( menu_scaling(), ImGui::GetCurrentContext() );
        if ( !enabled->dialogIsOpen() )
            enabled->enable( false );
    }
    else
    {
        if ( !allowRemoval_ )
            allowRemoval_ = true;
    }
}

void ImGuiMenu::draw_mr_menu()
{
    // Mesh
    ProgressBar::setup( menu_scaling() );
    const auto& viewportParameters = viewer->viewport().getParameters();
    if ( drawCollapsingHeader_( "Main", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        draw_history_block_();
        float w = ImGui::GetContentRegionAvail().x;
        float p = ImGui::GetStyle().FramePadding.x;
        if ( ImGui::Button( "Load##Main", ImVec2( ( w - p ) / 2.f - p - ImGui::GetFrameHeight(), 0 ) ) )
        {
            auto filenames = openFilesDialog( { .filters = AllFilter | MeshLoad::getFilters() | PointsLoad::getFilters() | SceneLoad::getFilters() } );
            viewer->loadFiles( filenames );
        }
        ImGui::SameLine( 0, p );
        draw_open_recent_button_();
        ImGui::SameLine( 0, p );
        if ( ImGui::Button( "Load Dir##Main", ImVec2( ( w - p ) / 2.f, 0 ) ) )
        {
            auto openDir = RibbonSchemaHolder::schema().items.find( "Open directory" );
            if ( openDir != RibbonSchemaHolder::schema().items.end() && openDir->second.item )
            {
                openDir->second.item->action();
            }
        }

        if ( ImGui::Button( "Save##Main", ImVec2( ( w - p ) / 2.f, 0 ) ) )
        {
            auto filters = MeshSave::getFilters() | LinesSave::getFilters() | PointsSave::getFilters()
#ifndef MRVIEWER_NO_VOXELS
                | VoxelsSave::getFilters()
#endif
            ;
            auto savePath = saveFileDialog( { .filters = filters } );
            if ( !savePath.empty() )
                viewer->saveToFile( savePath );
        }
        ImGui::SameLine( 0, p );

        if ( ImGui::Button( "Save Scene##Main", ImVec2( ( w - p ) / 2.f, 0 ) ) )
        {
            auto savePath = saveFileDialog( { .filters = SceneSave::getFilters() } );

            if ( !savePath.empty() )
                ProgressBar::orderWithMainThreadPostProcessing( "Saving scene", [savePath, &root = SceneRoot::get()]()->std::function<void()>
                {
                    auto res = ObjectSave::toAnySupportedSceneFormat( root, savePath, ProgressBar::callBackSetProgress );

                    return[savePath, res] ()
                    {
                        if ( res )
                            getViewerInstance().recentFilesStore().storeFile( savePath );
                        else
                            showError( "Error saving scene: " + res.error() );
                    };
                } );
        }

        if ( ImGui::Button( "New Issue##Main", ImVec2( w, 0 ) ) )
        {
            OpenLink( "https://meshinspector.github.io/ReportIssue" );
        }
        if ( ImGui::Button( "Capture Screen##Main", ImVec2( w, 0 ) ) )
        {
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t( now );
            auto name = fmt::format( "Screenshot_{:%Y-%m-%d_%H-%M-%S}", LocaltimeOrZero( t ) );

            auto savePath = saveFileDialog( {
                .fileName = name,
                .filters = ImageSave::getFilters(),
            } );
            if ( !savePath.empty() )
            {
                auto image = viewer->captureSceneScreenShot();
                auto res = ImageSave::toAnySupportedFormat( image, savePath );

                if ( !res.has_value() )
                    spdlog::warn( "Error saving screenshot: {}", res.error() );
            }
        }
    }

    // Viewing options
    if ( drawCollapsingHeader_( "Viewing Options", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        ImGui::PushItemWidth( 80 * menu_scaling() );
        auto fov = viewportParameters.cameraViewAngle;
        UI::drag<AngleUnit>( "Camera FOV", fov, 0.001f, 0.01f, 179.99f, { .sourceUnit = AngleUnit::degrees } );
        viewer->viewport().setCameraViewAngle( fov );

        bool showGlobalBasis = viewer->globalBasis->isVisible( viewer->viewport().id );
        ImGui::Checkbox( "Show Global Basis", &showGlobalBasis );
        viewer->viewport().showGlobalBasis( showGlobalBasis );

        bool showRotCenter = viewer->rotationSphere->isVisible( viewer->viewport().id );
        ImGui::Checkbox( "Show rotation center", &showRotCenter );
        viewer->viewport().showRotationCenter( showRotCenter );

        // Orthographic view
        bool orth = viewportParameters.orthographic;
        ImGui::Checkbox( "Orthographic view", &orth );
        viewer->viewport().setOrthographic( orth );

        static std::vector<std::string> shadingModes = { "Auto Detect", "Smooth", "Flat" };
        SceneSettings::ShadingMode shadingMode = SceneSettings::getDefaultShadingMode();
        ImGui::SetNextItemWidth( 120.0f * menu_scaling() );
        UI::combo( "Default Shading Mode", ( int* )&shadingMode, shadingModes );
        if ( shadingMode != SceneSettings::getDefaultShadingMode() )
            SceneSettings::setDefaultShadingMode( shadingMode );
        ImGui::PopItemWidth();

        bool showAxes = viewer->basisAxes->isVisible( viewer->viewport().id );
        ImGui::Checkbox( "Show axes", &showAxes );
        viewer->viewport().showAxes( showAxes );

        const std::string typeColorEditStr = "Background";
        auto backgroundColor = getStoredColor_( typeColorEditStr, viewportParameters.backgroundColor );
        if ( ImGui::ColorEdit4( typeColorEditStr.c_str(), &backgroundColor.x,
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel ) )
            storedColor_ = { typeColorEditStr,backgroundColor };
        else if ( !ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) && storedColor_ && storedColor_->first == typeColorEditStr )
            storedColor_ = {};
        viewer->viewport().setBackgroundColor( Color( backgroundColor ) );
    }

    if ( ImGui::Button( "Fit Data", ImVec2( -1, 0 ) ) )
    {
        viewer->viewport().preciseFitDataToScreenBorder( { 0.9f, false, FitMode::Visible } );
    }
    if ( ImGui::Button( "Fit Selected", ImVec2( -1, 0 ) ) )
    {
        viewer->viewport().preciseFitDataToScreenBorder( { 0.9f, false, FitMode::SelectedPrimitives } );
    }

    if ( viewer->isAlphaSortAvailable() )
    {
        bool alphaSortBackUp = viewer->isAlphaSortEnabled();
        bool alphaBoxVal = alphaSortBackUp;
        ImGui::Checkbox( "Alpha Sort", &alphaBoxVal );
        if ( alphaBoxVal != alphaSortBackUp )
            viewer->enableAlphaSort( alphaBoxVal );
    }

    if ( drawCollapsingHeader_( "Viewports" ) )
    {
        auto configBackup = viewportConfig_;
        ImGui::RadioButton( "Single", ( int* )&viewportConfig_, ViewportConfigurations::Single );
        ImGui::RadioButton( "Horizontal", ( int* )&viewportConfig_, ViewportConfigurations::Horizontal );
        ImGui::RadioButton( "Vertical", ( int* )&viewportConfig_, ViewportConfigurations::Vertical );
        ImGui::RadioButton( "Quad", ( int* )&viewportConfig_, ViewportConfigurations::Quad );
        if ( configBackup != viewportConfig_ )
        {
            for ( int i = int( viewer->viewport_list.size() ) - 1; i > 0; --i )
                viewer->erase_viewport( i );

            auto win = glfwGetCurrentContext();
            int window_width, window_height;
            glfwGetFramebufferSize( win, &window_width, &window_height );

            auto bounds = viewer->getViewportsBounds();

            float width = MR::width( bounds );
            float height = MR::height( bounds );

            ViewportRectangle rect;
            switch ( viewportConfig_ )
            {
            case Vertical:
                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height;
                viewer->viewport().setViewportRect( rect );

                rect.min.x = bounds.min.x + width * 0.5f;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height;
                viewer->append_viewport( rect );
                break;
            case Horizontal:
                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->viewport().setViewportRect( rect );

                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y + height * 0.5f;
                rect.max.x = rect.min.x + width;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->append_viewport( rect );
                break;
            case Quad:
                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->viewport().setViewportRect( rect );

                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y + height * 0.5f;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->append_viewport( rect );

                rect.min.x = bounds.min.x + width * 0.5f;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->append_viewport( rect );

                rect.min.x = bounds.min.x + width * 0.5f;
                rect.min.y = bounds.min.y + height * 0.5f;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->append_viewport( rect );
                break;
            case Single:
            default:
                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width;
                rect.max.y = rect.min.y + height;
                viewer->viewport().setViewportRect( rect );
                break;
            }
            postResize_( window_width, window_height );
        }
    }

    if ( drawCollapsingHeader_( "Clipping plane" ) )
    {
        auto plane = viewportParameters.clippingPlane;
        auto showPlane = viewer->clippingPlaneObject->isVisible( viewer->viewport().id );
        plane.n = plane.n.normalized();
        auto w = ImGui::GetContentRegionAvail().x;
        ImGui::SetNextItemWidth( w );
        UI::drag<NoUnit>( "##ClippingPlaneNormal", plane.n, 1e-3f );
        ImGui::SetNextItemWidth( w / 2.0f );
        UI::drag<NoUnit>( "##ClippingPlaneD", plane.d, 1e-3f );
        ImGui::SameLine();
        ImGui::Checkbox( "Show##ClippingPlane", &showPlane );
        viewer->viewport().setClippingPlane( plane );
        viewer->viewport().showClippingPlane( showPlane );
    }
    ImGui::Text( "Current view: %d", viewer->viewport().id.value() );
    mainWindowPos_ = ImGui::GetWindowPos();
    mainWindowSize_ = ImGui::GetWindowSize();
}

void ImGuiMenu::draw_history_block_()
{
    auto historyStore = viewer->getGlobalHistoryStore();
    if ( !historyStore )
        return;
    auto backUpColorBtn = ImGui::GetStyle().Colors[ImGuiCol_Button];
    auto backUpColorBtnH = ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered];
    auto backUpColorBtnA = ImGui::GetStyle().Colors[ImGuiCol_ButtonActive];
    const auto& colorDis = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
    auto undos = historyStore->getNActions( 10u, HistoryAction::Type::Undo );
    auto redos = historyStore->getNActions( 10u, HistoryAction::Type::Redo );
    if ( undos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = colorDis;
    }
    float w = ImGui::GetContentRegionAvail().x;
    float p = ImGui::GetStyle().FramePadding.x;
    if ( ImGui::Button( "Undo##Main", ImVec2( ( w - p ) / 2.f - p - ImGui::GetFrameHeight(), 0 ) ) && !undos.empty() )
    {
        historyStore->undo();
    }
    ImGui::SameLine( 0, p );
    if ( ImGui::BeginCombo( "##UndoStack", "##UndoStack", ImGuiComboFlags_NoPreview ) )
    {
        if ( undos.empty() )
            ImGui::CloseCurrentPopup();
        const auto storedColor = ImGui::GetStyle().Colors[ImGuiCol_Header];
        ImGui::GetStyle().Colors[ImGuiCol_Header] = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
        for ( int i = 0; i < undos.size(); ++i )
        {
            if ( ImGui::Selectable( ( undos[i] + "##" + std::to_string( i ) ).c_str() ) )
            {
                for ( int j = 0; j <= i; ++j )
                    historyStore->undo();
            }
        }
        ImGui::GetStyle().Colors[ImGuiCol_Header] = storedColor;
        ImGui::EndCombo();
    }
    if ( redos.empty() && !undos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = colorDis;
    }
    else if ( !redos.empty() && undos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = backUpColorBtn;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = backUpColorBtnH;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = backUpColorBtnA;
    }
    ImGui::SameLine( 0, p );
    if ( ImGui::Button( "Redo##Main", ImVec2( ( w - p ) / 2.f - p - ImGui::GetFrameHeight(), 0 ) ) && !redos.empty() )
    {
        historyStore->redo();
    }
    ImGui::SameLine( 0, p );
    if ( ImGui::BeginCombo( "##RedoStack", "##RedoStack", ImGuiComboFlags_NoPreview ) )
    {
        if ( redos.empty() )
            ImGui::CloseCurrentPopup();
        const auto storedColor = ImGui::GetStyle().Colors[ImGuiCol_Header];
        ImGui::GetStyle().Colors[ImGuiCol_Header] = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
        for ( int i = 0; i < redos.size(); ++i )
        {
            if ( ImGui::Selectable( ( redos[i] + "##" + std::to_string( i ) ).c_str() ) )
            {
                for ( int j = 0; j <= i; ++j )
                    historyStore->redo();
            }
        }
        ImGui::GetStyle().Colors[ImGuiCol_Header] = storedColor;
        ImGui::EndCombo();
    }
    if ( redos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = backUpColorBtn;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = backUpColorBtnH;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = backUpColorBtnA;
    }
}

void ImGuiMenu::draw_open_recent_button_()
{
    if ( ImGui::BeginCombo( "##Recently Loaded", "##Recently Loaded", ImGuiComboFlags_NoPreview ) )
    {
        auto filenames = viewer->recentFilesStore().getStoredFiles();
        if ( filenames.empty() )
            ImGui::CloseCurrentPopup();
        const auto storedColor = ImGui::GetStyle().Colors[ImGuiCol_Header];
        ImGui::GetStyle().Colors[ImGuiCol_Header] = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
        for ( const auto& file : filenames )
        {
            if ( ImGui::Selectable( utf8string( file ).c_str() ) )
                viewer->loadFiles( std::vector<std::filesystem::path>( { file } ) );
        }
        ImGui::GetStyle().Colors[ImGuiCol_Header] = storedColor;
        ImGui::EndCombo();
    }
}

void ImGuiMenu::drawShortcutsWindow_()
{
    const auto& style = ImGui::GetStyle();
    const float hotkeysWindowWidth = 300 * menu_scaling();
    size_t numLines = 2;

    if ( shortcutManager_ )
        numLines += shortcutManager_->getShortcutList().size();

    const float hotkeysWindowHeight = ( style.WindowPadding.y * 2 + numLines * ( ImGui::GetTextLineHeight() + style.ItemSpacing.y ) );

    ImVec2 windowPos = ImGui::GetMousePos();
    windowPos.x = std::min( windowPos.x, Viewer::instanceRef().framebufferSize.x - hotkeysWindowWidth );
    windowPos.y = std::min( windowPos.y, Viewer::instanceRef().framebufferSize.y - hotkeysWindowHeight );

    ImGui::SetNextWindowPos( windowPos, ImGuiCond_Appearing );
    ImGui::SetNextWindowSize( ImVec2( hotkeysWindowWidth, hotkeysWindowHeight ) );
    ImGui::Begin( "HotKeys", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoFocusOnAppearing );

#pragma warning(push)
#if _MSC_VER >= 1937 // Visual Studio 2022 version 17.7
#pragma warning(disable: 5267) //definition of implicit copy constructor is deprecated because it has a user-provided destructor
#endif
    ImFont font = *ImGui::GetFont();
#pragma warning(pop)
    font.Scale = 1.2f;
    ImGui::PushFont( &font );
    ImGui::Text( "Hot Key List" );
    ImGui::PopFont();
    ImGui::NewLine();
    if ( shortcutManager_ )
    {
        const auto& shortcutsList = shortcutManager_->getShortcutList();
        for ( const auto& [key, category, name] : shortcutsList )
            ImGui::Text( "%s - %s", ShortcutManager::getKeyFullString( key ).c_str(), name.c_str() );
    }
    ImGui::End();
}

SelectedTypesMask ImGuiMenu::calcSelectedTypesMask( const std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    SelectedTypesMask res{};

    if ( selectedObjs.empty() )
        return res;

    for ( const auto& obj : selectedObjs )
    {
        if ( !obj )
        {
            continue;
        }
        else if ( obj->asType<ObjectMesh>() )
        {
            res |= SelectedTypesMask::ObjectMeshBit;
        }
        else if ( obj->asType<ObjectMeshHolder>() )
        {
            res |= SelectedTypesMask::ObjectMeshHolderBit;
        }
        else if ( obj->asType<ObjectLinesHolder>() )
        {
            res |= SelectedTypesMask::ObjectLinesHolderBit;
        }
        else if ( obj->asType<ObjectPointsHolder>() )
        {
            res |= SelectedTypesMask::ObjectPointsHolderBit;
        }
        else if ( obj->asType<ObjectLabel>() )
        {
            res |= SelectedTypesMask::ObjectLabelBit;
        }
        else if ( obj->asType<FeatureObject>() )
        {
            res |= SelectedTypesMask::ObjectFeatureBit;
        }
        else if ( obj->asType<MeasurementObject>() )
        {
            res |= SelectedTypesMask::ObjectMeasurementBit;
        }
        else
        {
            res |= SelectedTypesMask::ObjectBit;
        }
    }

    return res;
}

float ImGuiMenu::getSceneInfoItemWidth_( int itemCount )
{
    if ( itemCount == 0 )
        return 0;
    /// 100 is the widest label's size
    return ( ImGui::GetContentRegionAvail().x - 100.0f * menu_scaling() - ImGui::GetStyle().ItemInnerSpacing.x * ( itemCount - 1 ) ) / float( itemCount );
}

void ImGuiMenu::add_modifier( std::shared_ptr<MeshModifier> modifier )
{
    if ( modifier )
        modifiers_.push_back( modifier );
}

void ImGuiMenu::allowSceneReorder( bool allow )
{
    sceneObjectsList_->allowSceneReorder( allow );
}

void ImGuiMenu::allowObjectsRemoval( bool allow )
{
    allowRemoval_ = allow;
}

void ImGuiMenu::tryRenameSelectedObject()
{
    const auto& selected = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();
    if ( selected.size() != 1 )
        return;
    renameBuffer_ = selected[0]->name();
    showRenameModal_ = true;
}

void ImGuiMenu::setObjectTreeState( const Object* obj, bool open )
{
    sceneObjectsList_->setObjectTreeState( obj, open );
}

void ImGuiMenu::setShowShortcuts( bool val )
{
    showShortcuts_ = val;
}

bool ImGuiMenu::getShowShortcuts() const
{
    return showShortcuts_;
}

void ImGuiMenu::PluginsCache::validate( const std::vector<ViewerPlugin*>& viewerPlugins )
{
    // if same then cache is valid
    if ( viewerPlugins == allPlugins_ )
        return;

    allPlugins_ = viewerPlugins;

    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
        sortedCustomPlufins_[t] = {};
    for ( const auto& plugin : allPlugins_ )
    {
        StateBasePlugin* customPlugin = dynamic_cast< StateBasePlugin* >( plugin );
        if ( customPlugin )
            sortedCustomPlufins_[int( customPlugin->getTab() )].push_back( customPlugin );
    }
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        auto& tabPlugins = sortedCustomPlufins_[t];
        std::sort( tabPlugins.begin(), tabPlugins.end(), [] ( const auto& a, const auto& b )
        {
            return a->sortString() < b->sortString();
        } );
    }
}

StateBasePlugin* ImGuiMenu::PluginsCache::findEnabled() const
{
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        const auto& tabPlugins = sortedCustomPlufins_[t];
        for ( auto plug : tabPlugins )
            if ( plug->isEnabled() )
                return plug;
    }
    return nullptr;
}

const std::vector<StateBasePlugin*>& ImGuiMenu::PluginsCache::getTabPlugins( StatePluginTabs tab ) const
{
    return sortedCustomPlufins_[int( tab )];
}

void ImGuiMenu::UiRenderManagerImpl::preRenderViewport( ViewportId viewport )
{
    const auto& v = getViewerInstance().viewport( viewport );
    auto rect = v.getViewportRect();

    ImVec2 cornerA( rect.min.x, ImGui::GetIO().DisplaySize.y - rect.max.y );
    ImVec2 cornerB( rect.max.x, ImGui::GetIO().DisplaySize.y - rect.min.y );

    ImGui::GetBackgroundDrawList()->PushClipRect( cornerA, cornerB );
    ImGui::GetForegroundDrawList()->PushClipRect( cornerA, cornerB );
}

void ImGuiMenu::UiRenderManagerImpl::postRenderViewport( ViewportId viewport )
{
    (void)viewport;
    ImGui::GetBackgroundDrawList()->PopClipRect();
    ImGui::GetForegroundDrawList()->PopClipRect();
}

BasicUiRenderTask::BackwardPassParams ImGuiMenu::UiRenderManagerImpl::beginBackwardPass( ViewportId viewport, UiRenderParams::UiTaskList& tasks )
{
    const auto& menuPlugin = getViewerInstance().getMenuPlugin();
    menuPlugin->drawSceneUiSignal( menuPlugin->menu_scaling(), viewport, tasks );

    return { .consumedInteractions = ImGui::GetIO().WantCaptureMouse * BasicUiRenderTask::InteractionMask::mouseHover };
}

void ImGuiMenu::UiRenderManagerImpl::finishBackwardPass( const BasicUiRenderTask::BackwardPassParams& params )
{
    if ( ImGui::GetIO().WantCaptureMouse )
    {
        // Some other UI is hovered, but not ours.
        consumedInteractions = {};
    }
    else
    {
        // Our UI is hovered.
        consumedInteractions = params.consumedInteractions;
    }
}

bool ImGuiMenu::UiRenderManagerImpl::canConsumeEvent( BasicUiRenderTask::InteractionMask event ) const
{
    // Here we only force-unblock events if one of our widgets is hovered.
    return
        !bool( consumedInteractions & BasicUiRenderTask::InteractionMask::mouseHover ) ||
        bool( consumedInteractions & event );
}

} // end namespace
