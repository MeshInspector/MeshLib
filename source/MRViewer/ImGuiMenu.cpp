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
#include "MRImGuiMultiViewport.h"
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
#include "MRMesh/MRString.h"
#include "MRUIQualityControl.h"
#include "MRRibbonFontHolder.h"
#include "MRI18n.h"

#ifndef MRVIEWER_NO_VOXELS
#include "MRVoxels/MRObjectVoxels.h"
#include "MRVoxels/MRVoxelsSave.h"
#endif

#ifndef __EMSCRIPTEN__
#include <fmt/chrono.h>
#endif

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#include "MRPch/MRWinapi.h"

#include <bitset>

namespace
{
// Reserved keys block
using OrderedKeys = std::bitset<ImGuiKey_NamedKey_END>;

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
    _t( "Point Array Size" ),
    _t( "Line Array Size" ),
    _t( "Triangle Array Size" ),
    _t( "Point Elements Number" ),
    _t( "Line Elements Number" ),
    _t( "Triangle Elements Number" )
};

constexpr std::array<const char*, size_t( MR::Viewer::EventType::Count )> cEventCounterNames =
{
    _t( "Mouse Down" ),
    _t( "Mouse Up" ),
    _t( "Mouse Move" ),
    _t( "Mouse Scroll" ),
    _t( "Key Down" ),
    _t( "Key Up" ),
    _t( "Key Repeat" ),
    _t( "Char Pressed" )
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
#ifdef NDEBUG
        ImGui::GetIO().ConfigDebugHighlightIdConflicts = false;
#endif
        if ( _viewer->isMultiViewportAvailable() && _viewer->getLaunchParams().multiViewport )
            ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable multi viewports in ImGui
        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        style.FrameRounding = 5.0f;
        updateScaling();
        reloadFonts();

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

    // init emscripten resize, fullscreen, mouse scroll callback
    // may duplicate an existing resize callback (resizeEmsCanvas in MRViewer.cpp)
#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCallbacks( viewer->window, "#canvas" );
#endif
}

void reserveKeyEvent( ImGuiKey key )
{
    assert( key < getOrderedKeys().size() );
    getOrderedKeys()[key] = true;
}

void ImGuiMenu::startFrame()
{
    MR_TIMER;
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

    // checking for mouse or keyboard events
    // this will start drawing multiple frames without a swapping to render the interface elements without flickering
    bool needIncrement = false;
    if ( ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable && context_ )
    {
        const auto& eq = context_->InputEventsQueue;
        if ( !eq.empty() )
        {
            needIncrement = eq.back().Type == ImGuiInputEventType_MouseButton ||
                eq.back().Type == ImGuiInputEventType_MouseWheel ||
                eq.back().Type == ImGuiInputEventType_Key;

            // add focused event at the and if we switched from child viewport to main one
            // in order to keep focused state valid and also don't lose valid g.IO.MousePos
            int focused = glfwGetWindowAttrib( viewer->window, GLFW_FOCUSED );
            if ( focused )
            {
                for ( int i = int( eq.size() ) - 1; i >= 0; --i )
                {
                    if ( eq[i].Type != ImGuiInputEventType_Focus )
                        continue;
                    if ( !eq[i].AppFocused.Focused )
                        ImGui::GetIO().AddFocusEvent( true );
                    break;
                }
            }
        }
    }

    ImGui::NewFrame();
    UI::getDefaultWindowRectAllocator().invalidateClosedWindows();

    if ( needIncrement && context_->MouseViewport != ImGui::GetMainViewport() ) // needIncrement can be true only if ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable && context_
    {
        // drawing multiple frames without a swapping to render the interface elements without flickering
        viewer->incrementForceRedrawFrames( viewer->forceRedrawMinimumIncrementAfterEvents, true );
    }
}

void ImGuiMenu::finishFrame()
{
    MR_TIMER;
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

        if ( ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable )
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();

            if ( context_ )
            {
                for ( int i = 1; i < context_->Viewports.Size; ++i )
                {
                    const auto* vp = context_->Viewports[i];
                    // if non-main viewport will be deleted in following frames we force redraw of main frame
                    // to ensure that there is at least one frame when both removed viewport and main viewport renders the window
                    if ( vp->LastFrameActive < context_->FrameCount )
                    {
                        viewer->forceSwapOnFrame();
                        break;
                    }
                }

                if ( viewer->isCurrentFrameSwapping() )
                {
                    ImGui::RenderPlatformWindowsDefault();

                    // if in swapping frame new viewport appears deffer swapping for main viewport for one frame
                    // to ensure that there is at least one frame when both new viewport and main viewport renders the window
                    static int prevViewports = context_->Viewports.Size;
                    if ( context_->Viewports.Size > prevViewports )
                        viewer->incrementForceRedrawFrames( 1, true );
                    prevViewports = context_->Viewports.Size;
                }
            }
            glfwMakeContextCurrent( backup_current_context );
        }
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
    std::filesystem::path winDirPath = GetWindowsInstallDirectory();
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

void ImGuiMenu::loadFonts( int font_size )
{
#ifdef _WIN32
    if ( viewer->isGLInitialized() )
    {
        ImGuiIO& io = ImGui::GetIO();

        auto fontPath = getMenuFontPath();

        if ( !io.Fonts->AddFontFromFileTTF(
            utf8string( fontPath ).c_str(), float( font_size ) ) )
        {
            assert( false && "Failed to load font!" );
            spdlog::error( "Failed to load font from `{}`.", utf8string( fontPath ) );

            ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
                droid_sans_compressed_size, float( font_size ) );
        }
    }
    else
    {
        ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
            droid_sans_compressed_size, float( font_size ) );
    }
#else
    ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
        droid_sans_compressed_size, float( font_size ) );
    //TODO: expand for non-Windows systems
#endif
}

void ImGuiMenu::reloadFonts( int fontSize )
{
    ImGui::GetIO().Fonts->Clear();
    loadFonts( fontSize );
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
    updateScaling();
    reloadFonts();
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

void ImGuiMenu::postFocus_( bool focused )
{
    ImGui_ImplGlfw_WindowFocusCallback( viewer->window, focused );
#ifdef _WIN32
    if ( focused && ImGui::isMultiViewportEnabled() )
    {
        std::vector<GLFWwindow*> processedWindow;
        for ( ImGuiWindow* win : ImGui::GetCurrentContext()->Windows )
        {
            if ( !win->Viewport )
                continue;
            GLFWwindow* glfwWindow = ( GLFWwindow* )win->Viewport->PlatformHandle;
            if ( !glfwWindow || getViewerInstance().window == glfwWindow )
                continue;

            auto findIt = std::find( processedWindow.begin(), processedWindow.end(), glfwWindow );
            if ( findIt != processedWindow.end() )
                continue;

            processedWindow.push_back( glfwWindow );
            {
                HWND hwnd = glfwGetWin32Window( glfwWindow );
                SetWindowPos( hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE );
            }
        }
    }
#endif
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
    ImGui_ImplGlfw_CharCallback( viewer->window, key );
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
    drawLabelsWindow();

    drawViewerWindow();

    drawAdditionalWindows();   
}

void ImGuiMenu::drawViewerWindow()
{
    float menu_width = 180.f * UI::scale();
    ImGui::SetNextWindowPos( ImVec2( 0.0f, 0.0f ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowSize( ImVec2( 0.0f, 0.0f ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowSizeConstraints( ImVec2( menu_width, -1.0f ), ImVec2( menu_width, -1.0f ) );
    ImGui::Begin( "Viewer", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize );
    ImGui::PushItemWidth( ImGui::GetWindowWidth() * 0.4f );
    drawViewerWindowContent();
    ImGui::PopItemWidth();
    ImGui::End();
}

void ImGuiMenu::drawLabelsWindow()
{
  // Text labels
  ImGuiMV::SetNextWindowPosMainViewport(ImVec2(0,0), ImGuiCond_Always);
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
                       ImVec2( viewerCoord.x / pixelRatio_, viewerCoord.y / pixelRatio_ ),
                       color.getUInt32(),
                       &text[0], &text[0] + text.size(), 0.0f,
                       clipByViewport ? &clipRect : nullptr );
}

float ImGuiMenu::pixelRatio()
{
    int bufferSize[2];
    int windowSize[2];
    GLFWwindow* window = glfwGetCurrentContext();
    if ( window )
    {
        glfwGetFramebufferSize( window, &bufferSize[0], &bufferSize[1] );
        glfwGetWindowSize( window, &windowSize[0], &windowSize[1] );
        return (float) bufferSize[0] / (float) windowSize[0];
    }
    return 1.0f;
}

float ImGuiMenu::hidpiScaling()
{
    float xScale = 1.0f;
    float yScale = 1.0f;
#ifndef __EMSCRIPTEN__
    GLFWwindow* window = glfwGetCurrentContext();
    if ( window )
        glfwGetWindowContentScale( window, &xScale, &yScale );
#endif
    return 0.5f * ( xScale + yScale );
}

void ImGuiMenu::updateScaling()
{
    hidpiScale_ = hidpiScaling();
    pixelRatio_ = pixelRatio();


    float newScaling = userScaling_;
#ifdef __EMSCRIPTEN__
    newScaling *= float( emscripten_get_device_pixel_ratio() );
#elif defined __APPLE__
    newScaling *= pixelRatio_;
#else
    newScaling *= hidpiScale_ / pixelRatio_;
#endif

    UI::detail::setScale( newScaling ); // Send the menu scale to the UI.
}

float ImGuiMenu::menuScaling() const
{
    return UI::scale();
}

float ImGuiMenu::menu_scaling() const
{
    float newScaling = userScaling_;
#ifdef __EMSCRIPTEN__
    newScaling *= float( emscripten_get_device_pixel_ratio() );
#elif defined __APPLE__
    newScaling *= pixelRatio_;
#else
    newScaling *= hidpiScale_ / pixelRatio_;
#endif

    return newScaling;
}

void ImGuiMenu::setUserScaling( float scaling )
{
    scaling = std::clamp( scaling, 0.5f, 4.0f );
    if ( scaling == userScaling_ )
        return;
    userScaling_ = scaling;
    CommandLoop::appendCommand( [&] ()
    {
        auto scaling = menuScaling();
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
        const float fpsWindowWidth = 300 * UI::scale();
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
            ImGui::Text( "%s: %zu", _tr( cGLPrimitivesCounterNames[i] ), viewer->getLastFrameGLPrimitivesCount( Viewer::GLPrimitivesType( i ) ) );
        ImGui::Separator();
        for ( int i = 0; i<int( Viewer::EventType::Count ); ++i )
            ImGui::Text( "%s: %zu", _tr( cEventCounterNames[i] ), viewer->getEventsCount( Viewer::EventType( i ) ) );
        ImGui::Separator();
        auto glBufferSizeStr = bytesString( viewer->getStaticGLBufferSize() );
        ImGui::Text( "%s: %s", _tr( "GL memory buffer" ), glBufferSizeStr.c_str() );
        auto prevFrameTime = viewer->getPrevFrameDrawTimeMillisec();
        if ( prevFrameTime > frameTimeMillisecThreshold_ )
            ImGui::TextColored( ImVec4( 1.0f, 0.3f, 0.3f, 1.0f ), "%s: %.1f ms", _tr( "Previous frame time" ), prevFrameTime );
        else
            ImGui::Text( "%s: %.1f ms", _tr( "Previous frame time" ), prevFrameTime );
        ImGui::Text( "%s: %zu", _tr( "Total frames" ), viewer->getTotalFrames() );
        ImGui::Text( "%s: %zu", _tr( "Swapped frames" ), viewer->getSwappedFrames() );
        ImGui::Text( "%s: %zu", _tr( "FPS" ), viewer->getFPS() );

        if ( UI::buttonCommonSize( _tr( "Reset" ), Vector2f( -1, 0 ) ) )
        {
            viewer->resetAllCounters();
        }
        if ( UI::buttonCommonSize( _tr( "Print Time to Log" ), Vector2f( -1, 0 ) ) )
        {
            printTimingTree();
            ProgressBar::printTimingTree();
        }
        ImGui::End();
    }

    if ( showRenameModal_ )
    {
        showRenameModal_ = false;
        ImGui::OpenPopup( "Rename object##rename" );
        popUpRenameBuffer_ = renameBuffer_;
    }

    ModalDialog renameDialog( "Rename object##rename", {
        .headline = _tr( "Rename Object" ),
        .closeOnClickOutside = true,
    } );
    if ( renameDialog.beginPopup() )
    {
        const auto& obj = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>().front();
        if ( !obj )
        {
            ImGui::CloseCurrentPopup();
        }
        if ( ImGui::IsWindowAppearing() )
            ImGui::SetKeyboardFocusHere();

        const auto& style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cInputPadding * UI::scale() } );
        ImGui::SetNextItemWidth( renameDialog.windowWidth() - 2 * style.WindowPadding.x - style.ItemInnerSpacing.x - ImGui::CalcTextSize( _tr( "Name" ) ).x );
        UI::inputText( _tr( "Name" ), popUpRenameBuffer_, ImGuiInputTextFlags_AutoSelectAll );
        ImGui::PopStyleVar();

        const float btnWidth = cModalButtonWidth * UI::scale();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
        if ( UI::button( _tr( "Ok" ), Vector2f( btnWidth, 0 ), ImGuiKey_Enter ) )
        {
            AppendHistory( std::make_shared<ChangeNameAction>( "Rename object from modal dialog", obj ) );
            obj->setName( popUpRenameBuffer_ );
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        ImGui::SetCursorPosX( renameDialog.windowWidth() - btnWidth - style.WindowPadding.x );
        if ( UI::button( _tr( "Cancel" ), Vector2f( btnWidth, 0 ), ImGuiKey_Escape ) )
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleVar();

        renameDialog.endPopup();
    }

    if ( showEditTag_ )
    {
        ImGui::OpenPopup( "Edit tag##edittag" );
        showEditTag_ = false;
    }

    ModalDialog editTagDialog( "Edit tag##edittag", {
        .headline = _tr( "Edit Tag" ),
        .closeButton = true,
        //.closeOnClickOutside = true, // FIXME: color picker closes the modal dialog on exit
    } );
    if ( editTagDialog.beginPopup() )
    {
        if ( ImGui::IsWindowAppearing() )
            ImGui::SetKeyboardFocusHere();

        const auto& style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cInputPadding * UI::scale() } );
        ImGui::SetNextItemWidth( editTagDialog.windowWidth() - 2 * style.WindowPadding.x - style.ItemInnerSpacing.x - ImGui::CalcTextSize( _tr( "Name" ) ).x );
        UI::inputText( _tr( "Name" ), tagEditorState_.name, ImGuiInputTextFlags_AutoSelectAll );
        ImGui::PopStyleVar();

        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cCheckboxPadding * UI::scale() } );
        UI::checkbox( _tr( "Assign Color" ), &tagEditorState_.hasFrontColor );
        ImGui::PopStyleVar();

        if ( tagEditorState_.hasFrontColor )
        {
            ImGui::ColorEdit4( _tr( "Selected Color" ), (float*)&tagEditorState_.selectedColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel );
            ImGui::ColorEdit4( _tr( "Unselected Color" ), (float*)&tagEditorState_.unselectedColor, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel );
        }

        const float btnWidth = cModalButtonWidth * UI::scale();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
        if ( UI::button( _tr( "Save" ), Vector2f( btnWidth, 0 ), ImGuiKey_Enter ) )
        {
            if ( const auto name = std::string{ trim( tagEditorState_.name ) }; !name.empty() && name != tagEditorState_.initName )
            {
                if ( tagEditorState_.hasFrontColor )
                {
                    VisualObjectTagManager::unregisterTag( tagEditorState_.initName );
                    VisualObjectTagManager::registerTag( name, {
                        .selectedColor = tagEditorState_.selectedColor,
                        .unselectedColor = tagEditorState_.unselectedColor,
                    } );
                }

                for ( auto obj : getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
                {
                    if ( obj->tags().contains( tagEditorState_.initName ) )
                    {
                        obj->removeTag( tagEditorState_.initName );
                        obj->addTag( name );
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
        if ( UI::button( _tr( "Cancel" ), Vector2f( btnWidth, 0 ), ImGuiKey_Escape ) )
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleVar();

        editTagDialog.endPopup();
    }

    drawModalMessage_();
}

void ImGuiMenu::expandObjectTreeAndScroll( const Object* obj )
{
    if ( sceneObjectsList_ )
        sceneObjectsList_->expandObjectTreeAndScroll( obj );
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

bool ImGuiMenu::simulateNameTagClickWithKeyboardModifiers( Object& object )
{
    return simulateNameTagClick( object, ImGui::IsKeyDown( UI::getImGuiModPrimaryCtrl() ) ? ImGuiMenu::NameTagSelectionMode::toggle : ImGuiMenu::NameTagSelectionMode::selectOne );
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

    std::string titleKey;
    std::string titleDisplay;
    if ( modalMessageType_ == NotificationType::Error )
    {
        titleKey = "Error";
        titleDisplay = s_tr( "Error" );
    }
    else if ( modalMessageType_ == NotificationType::Warning )
    {
        titleKey = "Warning";
        titleDisplay = s_tr( "Warning" );
    }
    else //if ( modalMessageType_ == MessageType::Info )
    {
        titleKey = "Info";
        titleDisplay = s_tr( "Info" );
    }

    const std::string titleImGui = " " + titleKey + "##modal";

    if ( showInfoModal_ &&
        !ImGui::IsPopupOpen( " Error##modal" ) && !ImGui::IsPopupOpen( " Warning##modal" ) && !ImGui::IsPopupOpen( " Info##modal" ) )
    {
        ImGui::OpenPopup( titleImGui.c_str() );
        showInfoModal_ = false;
    }

    ModalDialog modal( titleImGui, {
        .headline = titleDisplay,
        .text = storedModalMessage_,
        .closeOnClickOutside = true,
    } );
    if ( modal.beginPopup() )
    {
        const auto style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * UI::scale() } );
        if ( UI::button( _tr( "Okay" ), Vector2f( -1, 0 ), ImGuiKey_Enter ) )
            ImGui::CloseCurrentPopup();
        ImGui::PopStyleVar();

        modal.endPopup();
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

    // focus main window
    if ( ImGui::isMultiViewportEnabled() )
        glfwFocusWindow( getViewerInstance().window );
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
    ImGui::SetNextWindowPos( ImVec2( 180 * UI::scale(), 0 ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowSize( ImVec2( 230 * UI::scale(), 300 * UI::scale() ), ImGuiCond_FirstUseEver );
    ImGui::Begin(
        "Scene", nullptr
    );
    sceneObjectsList_->draw( -1 );

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
            _tr( "Selection Properties" ), nullptr,
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

    if ( allHaveVisualisation && drawCollapsingHeader_( _tr( "Draw Options" ) ) )
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

    if ( !drawCollapsingHeader_( _tr( "Information" ), ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowOverlap ) || selectedObjs.empty() )
        return resultingHeight();

    // draw World/Local toggles
    {
        auto pos = ImGui::GetCursorPos();
        pos.x += ImGui::GetContentRegionAvail().x + style.WindowPadding.x * 0.5f - style.FramePadding.x;
        pos.y -= ImGui::GetFrameHeightWithSpacing();
        const auto frameHeight = ImGui::GetFrameHeight();

        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 8.f * UI::scale(), 3.f * UI::scale() } );
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, style.ItemInnerSpacing );
        RibbonFontHolder iconsFont( RibbonFontManager::FontType::SemiBold, 0.75f );
        const auto worldText = s_tr( "WORLD" );
        const auto localText = s_tr( "LOCAL" );
        const auto worldTextSize = ImGui::CalcTextSize( worldText.c_str() );
        const auto localTextSize = ImGui::CalcTextSize( localText.c_str() );
        const ImVec2 layoutSize {
            worldTextSize.x + localTextSize.x + style.ItemSpacing.x + style.FramePadding.x * 4,
            std::max( worldTextSize.y, localTextSize.y ) + style.FramePadding.y * 2,
        };

        // draw invisible button to prevent misclicking the header
        ImGui::SetCursorPos( { pos.x - layoutSize.x - style.ItemSpacing.x, pos.y } );
        ImGui::SetNextItemAllowOverlap();
        ImGui::InvisibleButton( "##CoordToggleBackground", { layoutSize.x + style.ItemSpacing.x * 2, frameHeight } );

        pos.x -= layoutSize.x;
        pos.y += ( frameHeight - layoutSize.y ) / 2;
        ImGui::SetCursorPos( pos );

        auto showToggleButton = [&] ( const char* label, CoordType coordType )
        {
            const auto enabled = coordType_ == coordType;
            if ( enabled )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, Color::white() );
                ImGui::PushStyleColor( ImGuiCol_Button, style.Colors[ImGuiCol_ButtonActive] );
            }
            else
            {
                if ( ColorTheme::getPreset() == ColorTheme::Preset::Dark )
                {
                    ImGui::PushStyleColor( ImGuiCol_Button, Color::black() * .20f );
                    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, Color::black() * .30f );
                    ImGui::PushStyleColor( ImGuiCol_ButtonActive, Color::black() * .30f );
                }
                else
                {
                    ImGui::PushStyleColor( ImGuiCol_Button, Color::black() * .10f );
                    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, Color::black() * .05f );
                    ImGui::PushStyleColor( ImGuiCol_ButtonActive, Color::black() * .05f );
                }
            }

            if ( ImGui::Button( label ) )
                coordType_ = coordType;

            ImGui::PopStyleColor( enabled ? 2 : 3 );
        };
        showToggleButton( worldText.c_str(), CoordType::World );
        ImGui::SameLine();
        showToggleButton( localText.c_str(), CoordType::Local );

        iconsFont.popFont();
        ImGui::PopStyleVar( 2 );
    }

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
    selectionLocalBox_ = {};
    selectionWorldBox_ = {};
    std::optional<AffineXf3f> worldXf;
    bool showLocalBox = true;

    for ( const auto& obj : selectedObjs )
    {
        // compute units based on current coord type
        float lengthScale{}, areaScale{}, volumeScale{};
        switch ( coordType_ )
        {
        case CoordType::Local:
            lengthScale = 1.f;
            areaScale = 1.f;
            volumeScale = 1.f;
            break;

        case CoordType::World:
        {
            const auto xf = obj->worldXf();
            Matrix3f q, r;
            decomposeMatrix3( xf.A, q, r );
            const Vector3f scale{ r.x.x, r.y.y, r.z.z };
            lengthScale = ( scale.x + scale.y + scale.z ) / 3; // correct for uniform scales only
            areaScale = sqr( lengthScale );
            volumeScale = scale.x * scale.y * scale.z; // correct for not-uniform scales as well
        }
            break;
        }

        // Scene info update
        if ( auto vObj = obj->asType<VisualObject>() )
        {
            if ( auto box = vObj->getBoundingBox(); box.valid() )
                selectionLocalBox_.include( box );
            if ( auto box = vObj->getWorldBox(); box.valid() )
                selectionWorldBox_.include( box );
            if ( !worldXf )
                worldXf = vObj->worldXf();
            else if ( *worldXf != vObj->worldXf() )
                showLocalBox = false;
        }
        // Compute bounding box of group
        else if ( selectedObjs.size() == 1 )
        {
            for ( const auto& child : getAllObjectsInTree<VisualObject>( *obj, ObjectSelectivityType::Selectable ) )
            {
                if ( auto box = child->getBoundingBox(); box.valid() )
                    selectionLocalBox_.include( box );
                if ( auto box = child->getWorldBox(); box.valid() )
                    selectionWorldBox_.include( box );
                if ( !worldXf )
                    worldXf = child->worldXf();
                else if ( *worldXf != child->worldXf() )
                    showLocalBox = false;
            }
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
                totalVolume += volumeScale * mObj->volume();
                totalArea += areaScale * mObj->totalArea();
                totalSelectedArea += areaScale * mObj->selectedArea();
                avgEdgeLen = lengthScale * mObj->avgEdgeLen();
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
                totalLength += lengthScale * lObj->totalLength();
                avgEdgeLen = lengthScale * lObj->avgEdgeLen();
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

    ImGui::PushStyleVar( ImGuiStyleVar_ScrollbarSize, 12.0f );
    MR_FINALLY{ ImGui::PopStyleVar(); };

    const float smallItemSpacingY = std::round( 0.25f * cDefaultItemSpacing * UI::scale() );
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
        if ( !UI::inputTextCentered( _tr( "Object Name" ), renameBuffer_, getSceneInfoItemWidth_(), ImGuiInputTextFlags_AutoSelectAll ) )
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

            if ( UI::inputText( _tr( "Label" ), oldLabelParams_.labelBuffer, ImGuiInputTextFlags_AutoSelectAll ) )
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

    drawTagInformation_( selectedObjs );

    // customize input text widget design
    const ImVec4 originalFrameBgColor = ImGui::GetStyleColorVec4( ImGuiCol_FrameBg );
    const float originalFrameBorderSize = ImGui::GetStyle().FrameBorderSize;
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, Vector2f { 3.f, 3.f } * UI::scale() );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0 );
    ImGui::PushStyleColor( ImGuiCol_FrameBg, ImGui::GetStyleColorVec4( ImGuiCol_WindowBg ) );
    MR_FINALLY { ImGui::PopStyleVar( 2 ); ImGui::PopStyleColor( 1 ); };

    const float itemWidth = getSceneInfoItemWidth_( 3 ) * 3 + ImGui::GetStyle().ItemInnerSpacing.x * 2;

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
                UI::setTooltipIfHovered( _tr( "Selected / Total" ) );
        }
    };

    auto drawUnitInfo = [&] <class Units> ( const char* label, auto&& value, Units )
    {
        ImGui::SetNextItemWidth( itemWidth );
        UI::readOnlyValue<Units>( label, value, textColor, {}, labelColor );
    };

    auto drawDimensionsVec3 = [&] <class Units> ( const char* label, auto&& value, Units, std::optional<ImVec4> valueColor = {} )
    {
        ImGui::SetNextItemWidth( getSceneInfoItemWidth_() );
        UI::readOnlyValue<Units>( label, value, valueColor ? *valueColor : textColor, {}, labelColor );
    };

    if ( selectedObjs.size() == 1 )
    {
        UI::inputTextCenteredReadOnly( _tr( "Object Type" ), selectedObjs.front()->className(), itemWidth, textColor, labelColor );
    }
    else if ( selectedObjs.size() > 1 )
    {
        drawPrimitivesInfo( _tr( "Objects" ), selectedObjs.size() );
    }

    // Bounding box.
    if ( selectionLocalBox_.valid() && !( selectedObjs.size() == 1 && selectedObjs.front()->asType<FeatureObject>() ) )
    {
        ImGui::Spacing();
        ImGui::Spacing();

        RibbonFontHolder boldFont( RibbonFontManager::FontType::SemiBold, 1.f, false );

        switch ( coordType_ )
        {
        case CoordType::Local:
            if ( showLocalBox )
            {
                boldFont.pushFont();
                drawDimensionsVec3( _tr( "Local Box Size" ), selectionLocalBox_.size(), LengthUnit{}, labelColor );
                boldFont.popFont();
                UI::setTooltipIfHovered( _tr( "The edges of the tight axis-aligned bounding box in the local object space." ) );

                drawDimensionsVec3( _tr( "Local Box Min" ), selectionLocalBox_.min, LengthUnit{} );
                UI::setTooltipIfHovered( _tr( "Lower left corner of the tight axis-aligned bounding box in the local object space." ) );

                drawDimensionsVec3( _tr( "Local Box Max" ), selectionLocalBox_.max, LengthUnit{} );
                UI::setTooltipIfHovered( _tr( "Upper right corner of the tight axis-aligned bounding box in the local object space." ) );
            }
            break;

        case CoordType::World:
            boldFont.pushFont();
            drawDimensionsVec3( _tr( "World Box Size" ), selectionWorldBox_.size(), LengthUnit{}, labelColor );
            boldFont.popFont();
            UI::setTooltipIfHovered( _tr( "The edges of the tight axis-aligned bounding box in the world space." ) );

            drawDimensionsVec3( _tr( "World Box Min" ), selectionWorldBox_.min, LengthUnit{} );
            UI::setTooltipIfHovered( _tr( "Lower left corner of the tight axis-aligned bounding box in the world space." ) );

            drawDimensionsVec3( _tr( "World Box Max" ), selectionWorldBox_.max, LengthUnit{} );
            UI::setTooltipIfHovered( _tr( "Upper right corner of the tight axis-aligned bounding box in the world space." ) );

            break;
        }
    }

    if ( totalFaces || totalVerts || totalEdges || totalPoints )
    {
        ImGui::Spacing();
        ImGui::Spacing();

        drawPrimitivesInfo( _tr( "Triangles" ), totalFaces, totalSelectedFaces );
        drawPrimitivesInfo( _tr( "Vertices" ), totalVerts );
        drawPrimitivesInfo( _tr( "Edges" ), totalEdges, totalSelectedEdges );
        drawPrimitivesInfo( _tr( "Points" ), totalPoints, totalSelectedPoints );
    }

    if ( selectedObjs.size() == 1 && totalPoints )
        UI::inputTextCenteredReadOnly( _tr( "Point Normals" ), pointsHaveNormals ? _tr( "Yes" ) : _tr( "No" ), itemWidth, textColor, labelColor );

    if ( totalFaces )
    {
        drawUnitInfo( _tr( "Volume" ), totalVolume, VolumeUnit{} );
        UI::setTooltipIfHovered( _tr( "The volume surrounded by the mesh(es) in the world space." ) );

        ImGui::SetNextItemWidth( itemWidth );
        if ( totalSelectedArea > 0 )
        {
            UI::readOnlyValue<AreaUnit>( _tr( "Area" ), totalArea, selectedTextColor,
                { .decorationFormatString = valueToString<AreaUnit>( totalSelectedArea ) + " / {}" }, labelColor );
            UI::setTooltipIfHovered( _tr( "Selected / Total surface area in the world space." ) );
        }
        else
        {
            UI::readOnlyValue<AreaUnit>( _tr( "Area" ), totalArea, textColor, {}, labelColor );
            UI::setTooltipIfHovered( _tr( "Total surface area in the world space." ) );
        }
    }

    if ( totalLength > 0 )
    {
        drawUnitInfo( _tr( "Length" ), totalLength, LengthUnit{} );
        UI::setTooltipIfHovered( _tr( "The length of the lines in the world space." ) );
    }

    if ( selectedObjs.size() == 1 && avgEdgeLen > 0 )
    {
        drawUnitInfo( _tr( "Avg Edge Length" ), avgEdgeLen, LengthUnit{} );
        UI::setTooltipIfHovered( _tr( "Average edge length of the object(s) in the world space." ) );
    }

    drawPrimitivesInfo( _tr( "Holes" ), holes );
    drawPrimitivesInfo( _tr( "Components" ), components );

#ifndef MRVIEWER_NO_VOXELS
    if ( selectedObjs.size() == 1 && selectedObjs.front()->asType<ObjectVoxels>() )
    {
        ImGui::Spacing();
        ImGui::Spacing();

        if ( isValidVoxelsInfo( voxelDims ) )
            drawDimensionsVec3( _tr( "Voxels Dims" ), *voxelDims, NoUnit{} );
        if ( isValidVoxelsInfo( voxelSize ) )
            drawDimensionsVec3( _tr( "Voxel Size" ), *voxelSize, LengthUnit{} );
        if ( isValidVoxelsInfo( voxelActiveBox ) )
        {
            if ( voxelDims && ( voxelActiveBox->min != Vector3i{} || voxelActiveBox->max != voxelDims ) )
            {
                drawDimensionsVec3( _tr( "Active Box Min" ), voxelActiveBox->min, NoUnit{} );
                drawDimensionsVec3( _tr( "Active Box Max" ), voxelActiveBox->max, NoUnit{} );
            }
        }
        if ( voxelMinValue && voxelIsoValue && voxelMaxValue )
        {
            drawDimensionsVec3( _tr( "Min,Iso,Max" ), Vector3f{ *voxelMinValue, *voxelIsoValue, *voxelMaxValue }, NoUnit{} );
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
            // This is named either `Distance` or `Distance X`/Y/Z.
            drawUnitInfo( std::string( distance->getComparablePropertyName( 0 ) ).c_str(), distance->computeDistance(), LengthUnit{} );
            const auto delta = distance->getWorldDelta();
            drawDimensionsVec3( _tr( "X/Y/Z Distance" ), Vector3f{ std::abs( delta.x ), std::abs( delta.y ), std::abs( delta.z ) }, LengthUnit{} );
        }
        else if ( auto* angle = obj->asType<AngleMeasurementObject>() )
            drawUnitInfo( _tr( "Angle" ), angle->computeAngle(), AngleUnit{} );
        else if ( auto* radius = obj->asType<RadiusMeasurementObject>() )
            drawUnitInfo( radius->getDrawAsDiameter() ? _tr( "Diameter" ) : _tr( "Radius" ), radius->computeRadiusOrDiameter(), LengthUnit{} );
    }

    drawCustomSelectionInformation_( selectedObjs, {
        .textColor = textColor,
        .labelColor = labelColor,
        .selectedTextColor = selectedTextColor,
        .itemWidth = itemWidth,
        .item2Width = getSceneInfoItemWidth_( 2 ),
        .item3Width = getSceneInfoItemWidth_( 3 ),
    } );

    if ( selectedObjs.size() == 1 )
    {
        if ( auto comp = selectedObjs.front()->asType<ObjectComparableWithReference>() )
        {
            ImGui::Spacing();
            ImGui::Spacing();

            // Restore the original frame style for this.
            ImGui::PushStyleColor( ImGuiCol_FrameBg, originalFrameBgColor );
            MR_FINALLY{ ImGui::PopStyleColor(); };
            ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, originalFrameBorderSize );
            MR_FINALLY{ ImGui::PopStyleVar(); };

            drawComparablePropertiesEditor_( *comp );
        }
    }

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
                AppendHistory<ChangeXfAction>( _t( "Change Feature Transform" ), object );
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

void ImGuiMenu::drawComparablePropertiesEditor_( ObjectComparableWithReference& object )
{
    const float fullWidth = getSceneInfoItemWidth_( 1 );
    const ImVec2 buttonSize( ImGui::GetFrameHeight(), ImGui::GetFrameHeight() );

    const std::string_view notSpecifiedStr = "\xE2\x80\x94"; // U+2014 EM DASH

    // Reference values.
    const std::size_t numRefs = object.numComparisonReferenceValues();
    for ( std::size_t i = 0; i < numRefs; i++ )
    {
        auto nominalValue = object.getComparisonReferenceValue( i );

        ImGui::SetNextItemWidth( fullWidth );

        if ( std::visit( [&]( auto& elem ){ return UI::input<LengthUnit>( std::string( object.getComparisonReferenceValueName( i ) ).c_str(), elem, -FLT_MAX, FLT_MAX, { .decorationFormatString = nominalValue.isSet ? "{}" : notSpecifiedStr } ); }, nominalValue.var ) )
            object.setComparisonReferenceValue( i, nominalValue.var );

        if ( nominalValue.isSet )
        {
            ImGui::SameLine();

            ImGui::SetCursorPosX( ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - buttonSize.x );
            if ( UI::buttonEx( fmt::format( "\xC3\x97###removeNominal:{}", object.getComparisonReferenceValueName( i ) ).c_str(), buttonSize, { .customTexture = UI::getTexture( UI::TextureType::GradientBtnGray ).get() } ) ) // U+00D7 MULTIPLICATION SIGN
                object.setComparisonReferenceValue( i, {} );
        }
    }

    // Tolerances.
    const std::size_t numTols = object.numComparableProperties();
    for ( std::size_t i = 0; i < numTols; i++ )
    {
        std::string name;
        if ( numTols == 1 )
            name = s_tr( "Tolerance" );
        else
            name = fmt::format( "{} {}", object.getComparablePropertyName( i ), _tr( "tolerance" ) );

        ImGui::SetNextItemWidth( fullWidth );
        QualityControl::inputTolerance( name.c_str(), object, i );

        // The button to remove tolerance.
        if ( object.getComparisonTolerence( i ) )
        {
            ImGui::SameLine();

            ImGui::SetCursorPosX( ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - buttonSize.x );

            if ( UI::buttonEx( fmt::format( "\xC3\x97###removeTolerance:{}", name ).c_str(), buttonSize, { .customTexture = UI::getTexture( UI::TextureType::GradientBtnGray ).get() } ) ) // U+00D7 MULTIPLICATION SIGN
                object.setComparisonTolerance( i, {} );
        }
    }
}

bool ImGuiMenu::drawGeneralOptions( const std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    bool someChanges = false;
    const auto& selectedVisualObjs = SceneCache::getAllObjects<VisualObject, ObjectSelectivityType::Selected>();
    if ( !selectedVisualObjs.empty() )
    {
        const auto& viewportid = viewer->viewport().id;
        if ( make_visualize_checkbox( selectedVisualObjs, _tr( "Visibility" ), VisualizeMaskType::Visibility, viewportid ) )
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
    someChanges |= UI::checkboxMixed( _tr( "Lock Transform" ), &checked, mixedLocking );
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
    if ( !RibbonButtonDrawer::CustomCollapsingHeader( _tr( "Advanced" ) ) )
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
        make_visualize_checkbox( selectedObjs, _tr( "Polygon Offset" ), MeshVisualizePropertyType::PolygonOffsetFromCamera, viewportid );
        make_width<ObjectMeshHolder, float>( selectedObjs, _tr( "Point size" ), [&] ( const ObjectMeshHolder* objMesh )
        {
            return objMesh->getPointSize();
        }, [&] ( ObjectMeshHolder* objMesh, float value )
        {
            objMesh->setPointSize( value );
        } );
    }

    bool allIsObjLines = selectedMask == SelectedTypesMask::ObjectLinesHolderBit;
    if ( allIsObjLines )
    {
        make_width<ObjectLinesHolder, DashPattern>( selectedObjs, _tr( "Dash" ), [&] ( const ObjectLinesHolder* objLine )
        {
            return objLine->getDashPattern();
        }, [&] ( ObjectLinesHolder* objLine, const DashPattern& value )
        {
            objLine->setDashPattern( value );
        } );
    }

    make_light_strength( selectedObjs, _tr( "Shininess" ), [&] ( const VisualObject* obj )
    {
        return obj->getShininess();
    }, [&] ( VisualObject* obj, float value )
    {
        obj->setShininess( value );
    } );

    make_light_strength( selectedObjs, _tr( "Ambient Strength" ), [&] ( const VisualObject* obj )
    {
        return obj->getAmbientStrength();
    }, [&] ( VisualObject* obj, float value )
    {
        obj->setAmbientStrength( value );
    } );

    make_light_strength( selectedObjs, _tr( "Specular Strength" ), [&] ( const VisualObject* obj )
    {
        return obj->getSpecularStrength();
    }, [&] ( VisualObject* obj, float value )
    {
        obj->setSpecularStrength( value );
    } );

    bool allIsObjPoints = selectedMask == SelectedTypesMask::ObjectPointsHolderBit;

    if ( allIsObjPoints )
    {
        make_points_discretization( selectedObjs, _tr( "Visual Sampling" ), [&] ( const ObjectPointsHolder* data )
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



        make_slider<float, FeatureObject>( selectedFeatureObjs, _tr( "Point size" ),
            [&] ( const FeatureObject* data ){ return data->getPointSize(); },
            [&]( FeatureObject* data, float value ){ data->setPointSize( value ); }, minPointSize, maxPointSize );
        make_slider<float, FeatureObject>( selectedFeatureObjs, _tr( "Line width" ),
            [&] ( const FeatureObject* data ){ return data->getLineWidth(); },
            [&]( FeatureObject* data, float value ){ data->setLineWidth( value ); }, minLineWidth, maxLineWidth );

        make_slider<float, FeatureObject>( selectedFeatureObjs, _tr( "Point subfeatures size" ),
            [&] ( const FeatureObject* data ){ return data->getSubfeaturePointSize(); },
            [&]( FeatureObject* data, float value ){ data->setSubfeaturePointSize( value ); }, minPointSize, maxPointSize );
        make_slider<float, FeatureObject>( selectedFeatureObjs, _tr( "Line subfeatures width" ),
            [&] ( const FeatureObject* data ){ return data->getSubfeatureLineWidth(); },
            [&]( FeatureObject* data, float value ){ data->setSubfeatureLineWidth( value ); }, minLineWidth, maxLineWidth );

        make_slider<float, FeatureObject>( selectedFeatureObjs, _tr( "Main component alpha" ),
            [&] ( const FeatureObject* data ){ return data->getMainFeatureAlpha(); },
            [&]( FeatureObject* data, float value ){ data->setMainFeatureAlpha( value ); }, 0, 1 );
        make_slider<float, FeatureObject>( selectedFeatureObjs, _tr( "Point subfeatures alpha" ),
            [&] ( const FeatureObject* data ){ return data->getSubfeatureAlphaPoints(); },
            [&]( FeatureObject* data, float value ){ data->setSubfeatureAlphaPoints( value ); }, 0, 1 );
        make_slider<float, FeatureObject>( selectedFeatureObjs, _tr( "Line subfeatures alpha" ),
            [&] ( const FeatureObject* data ){ return data->getSubfeatureAlphaLines(); },
            [&]( FeatureObject* data, float value ){ data->setSubfeatureAlphaLines( value ); }, 0, 1 );
        make_slider<float, FeatureObject>( selectedFeatureObjs, _tr( "Mesh subfeatures alpha" ),
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
        UI::button( _tr( "Remove" ), Vector2f( -1, 0 ) ) :
        ImGui::Button( _tr( "Remove" ), ImVec2( -1, 0 ) );
    if ( clicked )
    {
        someChanges |= true;
        if ( allowRemoval_ )
        {
            SCOPED_HISTORY( _t( "Remove Objects (context)" ) );
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
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Shading" ), MeshVisualizePropertyType::EnableShading, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Flat Shading" ), MeshVisualizePropertyType::FlatShading, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Edges" ), MeshVisualizePropertyType::Edges, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Points" ), MeshVisualizePropertyType::Points, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Selected Edges" ), MeshVisualizePropertyType::SelectedEdges, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Selected Tri-s" ), MeshVisualizePropertyType::SelectedFaces, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Borders" ), MeshVisualizePropertyType::BordersHighlight, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Triangles" ), MeshVisualizePropertyType::Faces, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Transparency" ), MeshVisualizePropertyType::OnlyOddFragments, viewportid );
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
            someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Texture" ), MeshVisualizePropertyType::Texture, viewportid );
    }
    if ( allIsObjLines )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Points" ), LinesVisualizePropertyType::Points, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Smooth corners" ), LinesVisualizePropertyType::Smooth, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Dashed" ), LinesVisualizePropertyType::Dashed, viewportid );
        make_width<ObjectLinesHolder, float>( selectedVisualObjs, _tr( "Line width" ), [&] ( const ObjectLinesHolder* objLines )
        {
            return objLines->getLineWidth();
        }, [&] ( ObjectLinesHolder* objLines, float value )
        {
            objLines->setLineWidth( value );
        } );
        make_width<ObjectLinesHolder, float>( selectedVisualObjs, _tr( "Point size" ), [&] ( const ObjectLinesHolder* objLines )
        {
            return objLines->getPointSize();
        }, [&] ( ObjectLinesHolder* objLines, float value )
        {
            objLines->setPointSize( value );
        } );
    }
    if ( allIsObjPoints )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Selected Points" ), PointsVisualizePropertyType::SelectedVertices, viewportid );
        make_width<ObjectPointsHolder, float>( selectedVisualObjs, _tr( "Point size" ), [&] ( const ObjectPointsHolder* objPoints )
        {
            return objPoints->getPointSize();
        }, [&] ( ObjectPointsHolder* objPoints, float value )
        {
            objPoints->setPointSize( value );
        } );
    }
    if ( allIsObjLabels )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Always on top" ), VisualizeMaskType::DepthTest, viewportid, true );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Source point" ), LabelVisualizePropertyType::SourcePoint, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Background" ), LabelVisualizePropertyType::Background, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Contour" ), LabelVisualizePropertyType::Contour, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Leader line" ), LabelVisualizePropertyType::LeaderLine, viewportid );
    }
    if ( allIsFeatureObj )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Subfeatures" ), FeatureVisualizePropertyType::Subfeatures, viewportid );
    }
    someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Name" ), VisualizeMaskType::Name, viewportid );
    if ( allIsFeatureObj )
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Extra information next to name" ), FeatureVisualizePropertyType::DetailsOnNameTag, viewportid );
    if ( viewer->experimentalFeatures )
        someChanges |= make_visualize_checkbox( selectedVisualObjs, _tr( "Clipping" ), VisualizeMaskType::ClippedByPlane, viewportid );

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
        ImGui::SetNextItemWidth( 75.0f * UI::scale() );

        if (ImGui::BeginCombo( _tr( "Viewport Id" ),
            selectedViewport_.value() == 0 ? _tr( "Default" ) :
            std::to_string( selectedViewport_.value() ).c_str() ) )
        {
            if ( ImGui::Selectable( _tr( "Default" ) ) )
                selectedViewport_ = ViewportId{ 0 };

            for ( const auto& viewport : getViewerInstance().viewport_list )
            {
                if ( ImGui::Selectable( std::to_string( viewport.id.value() ).c_str() ) )
                    selectedViewport_ = viewport.id;
            }

            ImGui::EndCombo();
        }
    }

    make_color_selector<VisualObject>( selectedVisualObjs, (s_tr( "Selected color" ) + "##" + std::to_string(selectedViewport_.value())).c_str(), [&] ( const VisualObject* data )
    {
        return Vector4f( data->getFrontColor(true, selectedViewport_ ) );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setFrontColor( Color( color ), true, selectedViewport_ );
    } );
    make_color_selector<VisualObject>( selectedVisualObjs, _tr( "Unselected color" ), [&] ( const VisualObject* data )
    {
        return Vector4f( data->getFrontColor( false, selectedViewport_ ) );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setFrontColor( Color( color ), false, selectedViewport_ );
    } );
    make_color_selector<VisualObject>( selectedVisualObjs, _tr( "Back Triangles color" ), [&] ( const VisualObject* data )
    {
        return Vector4f( data->getBackColor( selectedViewport_ ) );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setBackColor( Color( color ), selectedViewport_ );
    } );

    if ( !selectedMeshObjs.empty() )
    {
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, _tr( "Edges color" ), [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getEdgesColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setEdgesColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, _tr( "Points color" ), [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getPointsColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setPointsColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, _tr( "Selected Tri-s color" ), [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getSelectedFacesColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setSelectedFacesColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, _tr( "Selected Edges color" ), [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getSelectedEdgesColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setSelectedEdgesColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, _tr( "Borders color" ), [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getBordersColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setBordersColor( Color( color ), selectedViewport_ );
        } );
    }
    if ( !selectedPointsObjs.empty() )
    {
        make_color_selector<ObjectPointsHolder>( selectedPointsObjs, _tr( "Selected Points color" ), [&] ( const ObjectPointsHolder* data )
        {
            return Vector4f( data->getSelectedVerticesColor( selectedViewport_ ) );
        }, [&] ( ObjectPointsHolder* data, const Vector4f& color )
        {
            data->setSelectedVerticesColor( Color( color ), selectedViewport_ );
        } );
    }
    if ( !selectedLabelObjs.empty() )
    {
        make_color_selector<ObjectLabel>( selectedLabelObjs, _tr( "Source point color" ), [&] ( const ObjectLabel* data )
        {
            return Vector4f( data->getSourcePointColor( selectedViewport_ ) );
        }, [&] ( ObjectLabel* data, const Vector4f& color )
        {
            data->setSourcePointColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectLabel>( selectedLabelObjs, _tr( "Leader line color" ), [&] ( const ObjectLabel* data )
        {
            return Vector4f( data->getLeaderLineColor( selectedViewport_ ) );
        }, [&] ( ObjectLabel* data, const Vector4f& color )
        {
            data->setLeaderLineColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectLabel>( selectedLabelObjs, _tr( "Contour color" ), [&] ( const ObjectLabel* data )
        {
            return Vector4f( data->getContourColor( selectedViewport_ ) );
        }, [&] ( ObjectLabel* data, const Vector4f& color )
        {
            data->setContourColor( Color( color ), selectedViewport_ );
        } );
    }

    if ( !selectedFeatureObjs.empty() )
    {
        make_color_selector<FeatureObject>( selectedFeatureObjs, _tr( "Decorations color (selected)" ), [&] ( const FeatureObject* data )
        {
            return Vector4f( data->getDecorationsColor( true, selectedViewport_ ) );
        }, [&] ( FeatureObject* data, const Vector4f& color )
        {
            data->setDecorationsColor( Color( color ), true, selectedViewport_ );
        } );

        make_color_selector<FeatureObject>( selectedFeatureObjs, _tr( "Decorations color (unselected)" ), [&] ( const FeatureObject* data )
        {
            return Vector4f( data->getDecorationsColor( false, selectedViewport_ ) );
        }, [&] ( FeatureObject* data, const Vector4f& color )
        {
            data->setDecorationsColor( Color( color ), false, selectedViewport_ );
        } );
    }

    if ( !selectedVisualObjs.empty() )
    {
        make_slider<std::uint8_t, VisualObject>( selectedVisualObjs, _tr( "Opacity" ), [&] ( const VisualObject* data )
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

void ImGuiMenu::drawTagInformation_( const std::vector<std::shared_ptr<Object>>& selected )
{
    const auto initWidth = ImGui::GetContentRegionAvail().x;
    const auto initCursorScreenPos = ImGui::GetCursorScreenPos();
    // using std::max to avoid assert in `ImGui::InvisibleButton` in degenerete window size scenario
    const auto itemWidth = std::max( 1.0f, getSceneInfoItemWidth_() );

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
        text += ( tagCount != 0 ? " + " : "" ) + fmt::format( f_tr( "{} uncommon tag", "{} uncommon tags", uncommonTagCount ), uncommonTagCount );
    if ( text.empty() )
        text = "–";

    auto textSize = ImGui::CalcTextSize( text.c_str() );
    if ( itemWidth < textSize.x )
    {
        // TODO: cache
        const auto ellipsisSize = ImGui::CalcTextSize( "..." );
        auto textLen = text.size();
        for ( --textLen; textLen > 0; --textLen )
        {
            textSize = ImGui::CalcTextSize( text.data(), text.data() + textLen );
            if ( textSize.x + ellipsisSize.x <= itemWidth )
                break;
        }
        text = text.substr( 0, textLen ) + "...";
        textSize = ImGui::CalcTextSize( text.c_str() );
    }

    const auto initCursorPos = ImGui::GetCursorPos();
    ImGui::SetNextItemAllowOverlap();
    UI::inputTextCentered( _tr( "Tags" ), text, itemWidth );

    ImGui::SetCursorPos( initCursorPos );
    if ( ImGui::InvisibleButton( "##EnterTagsWindow", { itemWidth, ImGui::GetFrameHeight() } ) )
        ImGui::OpenPopup( "TagsPopup" );

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

        const auto& style = ImGui::GetStyle();

        

        const auto buttonWidth = [&] ( const char* label )
        {
            return style.FramePadding.x * 2.f + ImGui::CalcTextSize( label, NULL, true ).x;
        };

        RibbonFontHolder iconsFont( RibbonFontManager::FontType::Icons, cDefaultFontSize / cBigIconSize );

        const auto* removeButtonText = iconsFont.isPushed() ? "\xef\x80\x8d" : "X";
        const auto* addButtonText = iconsFont.isPushed() ? "\xef\x81\x95" : "+";
        const auto removeButtonWidth = buttonWidth( removeButtonText );
        const auto addButtonWidth = buttonWidth( addButtonText );
        iconsFont.popFont();

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
                ImGui::PushStyleColor( ImGuiCol_Text, ImGui::getLuminance( color ) > 0.5f ? Color::black() : Color::white() );
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
                ImGui::PopStyleColor( 2 );

            ImGui::SameLine( initCursorPosX + buttonWidth( tag.c_str() ), 0 );

            iconsFont.pushFont();

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
            iconsFont.popFont();

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
        ImGui::SetNextItemWidth( ImGui::GetContentRegionAvail().x - style.ItemInnerSpacing.x - addButtonWidth );
        if ( ImGui::InputTextWithHint( "##TagNew", _tr( "Type to add new tag..." ), &tagNewName_, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion, tagCompletion, &allKnownTags ) )
        {
            if ( const auto name = std::string{ trim( tagNewName_ ) }; !name.empty() )
                for ( const auto& selObj : selected )
                    selObj->addTag( name );
            tagNewName_.clear();
        }

        iconsFont.pushFont();

        ImGui::SameLine( 0, style.ItemInnerSpacing.x );
        if ( ImGui::Button( addButtonText ) )
        {
            if ( const auto name = std::string{ trim( tagNewName_ ) }; !name.empty() )
                for ( const auto& selObj : selected )
                    selObj->addTag( name );
            tagNewName_.clear();
        }
        iconsFont.popFont();

        ImGui::EndPopup();
    }
    ImGui::PopStyleColor();
}

void ImGuiMenu::drawMixedTransformField_( const char* labelId, int columns, const char* trailingLabel )
{
    // Long-dash (em-dash) rendered in a read-only box per column; used when selected objects
    // have differing transforms, so the field communicates "values differ" and cannot be edited.
    static const std::string kDash = "\xE2\x80\x94";
    const float w = getSceneInfoItemWidth_( columns );
    for ( int i = 0; i < columns; ++i )
    {
        if ( i )
            ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
        const std::string label = ( i == columns - 1 && trailingLabel && *trailingLabel )
            ? fmt::format( "{}##{}{}", trailingLabel, labelId, i )
            : fmt::format( "##{}{}", labelId, i );
        UI::inputTextCenteredReadOnly( label.c_str(), kDash, w );
    }
}

std::shared_ptr<CombinedHistoryAction> ImGuiMenu::makeObjectsXfHistoryAction_(
    const std::string& name,
    const std::vector<std::shared_ptr<Object>>& objs ) const
{
    std::vector<std::shared_ptr<HistoryAction>> subs;
    subs.reserve( objs.size() );
    for ( const auto& o : objs )
        subs.push_back( std::make_shared<ChangeXfAction>( name, o ) );
    return std::make_shared<CombinedHistoryAction>( name, subs );
}

void ImGuiMenu::applyXfToObjects_(
    const AffineXf3f& xf,
    const std::vector<std::shared_ptr<Object>>& objs )
{
    for ( const auto& o : objs )
        o->setXf( xf );
}

float ImGuiMenu::drawTransform_()
{
    // Use topmost-selected objects so that when a parent and its child are both selected,
    // only the parent is edited — otherwise applying the same xf to both would move the
    // child twice (once through its own xf, once through the parent's cascade).
    const auto& selected = SceneCache::getAllTopmostObjects<Object, ObjectSelectivityType::Selected>();

    // Hide the panel if nothing is selected or any selected object is locked —
    // a generalisation of the original single-object `!selected[0]->isLocked()` gate.
    const bool canShow = !selected.empty()
        && std::none_of( selected.begin(), selected.end(),
            []( const auto& o ){ return o->isLocked(); } );
    if ( !canShow )
    {
        transformBlockShown_ = false;
        return 0.f;
    }

    auto& style = ImGui::GetStyle();

    // Fix up the scene-list scroll on the frame the transform block first appears, so the
    // selected row stays in view as the properties panel grows by the block's height.
    if ( !transformBlockShown_ )
    {
        transformBlockShown_ = true;
        sceneObjectsList_->setNextFrameFixScroll();
    }

    // When transforms differ across the selection, we display "—" in the fields, do not
    // enter the editing path, and do not open the context menu — its Copy/Paste/Reset
    // operations assume a single shared xf to read from and write back to.
    const AffineXf3f& xf0 = selected.front()->xf();
    const bool allSame = std::all_of( selected.begin() + 1, selected.end(),
        [&]( const auto& o ){ return o->xf() == xf0; } );

    float resultHeight_ = ImGui::GetTextLineHeight() + style.FramePadding.y * 2 + style.ItemSpacing.y;
    bool openedContext = false;
    if ( drawCollapsingHeaderTransform_() )
    {
        openedContext = drawTransformContextMenu_( selected );
        const float transformHeight = ( ImGui::GetTextLineHeight() + style.FramePadding.y * 2 ) * 3 + style.ItemSpacing.y * 2;
        resultHeight_ += transformHeight + style.ItemSpacing.y;
        ImGui::BeginChild( "SceneTransform", ImVec2( 0, transformHeight ) );

        auto xf = xf0;
        Matrix3f q, r;
        Vector3f euler;
        Vector3f scale;
        if ( allSame )
        {
            decomposeMatrix3( xf.A, q, r );
            euler = ( 180 / PI_F ) * q.toEulerAngles();
            scale = Vector3f{ r.x.x, r.y.y, r.z.z };
        }

        bool inputDeactivated = false;
        bool inputChanged = false;

        ImGui::PushItemWidth( getSceneInfoItemWidth_( 3 ) );
        if ( !allSame )
        {
            // Mixed: render "—" in the same column layout the drag widgets would use.
            drawMixedTransformField_( "scaleMixed", uniformScale_ ? 1 : 3 );
            ImGui::SameLine( 0, uniformScale_
                ? ImGui::GetStyle().ItemSpacing.x
                : ImGui::GetStyle().ItemInnerSpacing.x );
        }
        else if ( uniformScale_ )
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
        auto diff = ImGui::GetStyle().FramePadding.y - cCheckboxPadding * UI::scale();
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + diff );
        UI::checkbox( _tr( "Uni-scale" ), &uniformScale_ );
        window->DC.CursorPosPrevLine.y -= diff;
        UI::setTooltipIfHovered( _tr( "Selects between uniform scaling or separate scaling along each axis" ) );
        ImGui::PopItemWidth();

        if ( !allSame )
        {
            drawMixedTransformField_( "rotationMixed", 3, _tr( "Rotation XYZ" ) );
        }
        else
        {
            ImGui::SetNextItemWidth( getSceneInfoItemWidth_() );
            bool rotationChanged = UI::drag<AngleUnit>( _tr( "Rotation XYZ" ), euler, invertedRotation_ ? -0.1f : 0.1f, -360.f, 360.f, { .sourceUnit = AngleUnit::degrees } );
            bool rotationDeactivatedAfterEdit = ImGui::IsItemDeactivatedAfterEdit();
            ImGui::SetItemTooltip( "%s", _tr( "Rotation round [X, Y, Z] axes respectively." ) );
            inputChanged = inputChanged || rotationChanged;
            inputDeactivated = inputDeactivated || rotationDeactivatedAfterEdit;
            if ( ImGui::IsItemHovered() )
            {
                ImGui::BeginTooltip();
                ImGui::Text( "%s", _tr( "Sequential intrinsic rotations around Oz, Oy and Ox axes." ) ); // see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations
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
        }

        if ( !allSame )
        {
            drawMixedTransformField_( "translationMixed", 3, _tr( "Translation" ) );
        }
        else
        {
            const auto trSpeed = ( selectionLocalBox_.valid() && selectionLocalBox_.diagonal() > std::numeric_limits<float>::epsilon() ) ? 0.003f * selectionLocalBox_.diagonal() : 0.003f;

            ImGui::SetNextItemWidth( getSceneInfoItemWidth_() );
            auto wbsize = selectionWorldBox_.valid() ? selectionWorldBox_.size() : Vector3f::diagonal( 1.f );
            auto minSizeDim = wbsize.length();
            if ( minSizeDim == 0 )
                minSizeDim = 1.f;
            auto translation = xf.b;
            auto translationChanged = UI::drag<LengthUnit>( _tr( "Translation" ), translation, trSpeed, -cMaxTranslationMultiplier * minSizeDim, +cMaxTranslationMultiplier * minSizeDim );
            inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();

            if ( translationChanged )
                xf.b = translation;

            if ( xfHistUpdated_ )
                xfHistUpdated_ = !inputDeactivated;

            if ( xf != xf0 && !xfHistUpdated_ )
            {
                // One combined history entry per drag, regardless of how many objects
                // were selected — so Ctrl+Z reverts them all in one step.
                AppendHistory( makeObjectsXfHistoryAction_( _t( "Manual Change Transform" ), selected ) );
                xfHistUpdated_ = true;
            }
            applyXfToObjects_( xf, selected );
        }
        ImGui::EndChild();
        if ( !openedContext )
            openedContext = drawTransformContextMenu_( selected );
    }
    if ( !openedContext )
        drawTransformContextMenu_( selected );

    return resultHeight_;
}

bool ImGuiMenu::drawCollapsingHeader_( const char* label, ImGuiTreeNodeFlags flags )
{
    return ImGui::CollapsingHeader( label, flags );
}

bool ImGuiMenu::drawCollapsingHeaderTransform_()
{
    return drawCollapsingHeader_( _tr( "Transform" ), ImGuiTreeNodeFlags_DefaultOpen );
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
    ImGui::PushItemWidth( 40 * UI::scale() );
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

    ImGui::PushItemWidth( 50 * UI::scale() );
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

    ImGui::PushItemWidth( 100 * UI::scale() );
    UI::slider<NoUnit>( label, value, min, max );

    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( value != valueConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data.get(), T( value ) );
}

template<typename ObjType, typename ValueT>
void ImGuiMenu::make_width( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label,
    std::function<ValueT( const ObjType* )> getter,
    std::function<void( ObjType*, const ValueT& )> setter )
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
        value = ValueT{};
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }
    const auto valueConstForComparation = value;

    if constexpr ( std::is_same_v<ValueT, float> )
    {
        ImGui::PushItemWidth( 50 * menuScaling() );
        UI::drag<PixelSizeUnit>( label, value, 0.02f, 0.5f, 30.0f );
    }
    else
    {
        ImGui::PushItemWidth( 120 * menuScaling() );
        UI::drag<NoUnit>( label, value, 0.02f, uint8_t( 0 ), uint8_t( 50 ) );
    }
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

    ImGui::SetNextItemWidth( 50 * UI::scale() );
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

void ImGuiMenu::drawShortcutsWindow_()
{
    const auto& style = ImGui::GetStyle();
    const float hotkeysWindowWidth = 300 * UI::scale();
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

    ImGui::PushFont( nullptr, ImGui::GetStyle().FontSizeBase * 1.2f );
    ImGui::Text( "%s", _tr( "Hot Key List" ) );
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
    return ( ImGui::GetContentRegionAvail().x - 100.0f * UI::scale() - ImGui::GetStyle().ItemInnerSpacing.x * ( itemCount - 1 ) ) / float( itemCount );
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

void ImGuiMenu::UiRenderManagerImpl::preRenderViewport( ViewportId viewport )
{
    const auto& v = getViewerInstance().viewport( viewport );
    auto rect = v.getViewportRect();

    ImVec2 cornerA = ImGuiMV::Window2ScreenSpaceImVec2( ImVec2( rect.min.x, ImGui::GetIO().DisplaySize.y - rect.max.y ) );
    ImVec2 cornerB = ImGuiMV::Window2ScreenSpaceImVec2( ImVec2( rect.max.x, ImGui::GetIO().DisplaySize.y - rect.min.y ) );

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
    menuPlugin->drawSceneUiSignal( viewport, tasks );

    return {
        .consumedInteractions = ( ImGui::GetIO().WantCaptureMouse || getViewerInstance().getHoveredViewportIdOrInvalid() != viewport ) * BasicUiRenderTask::InteractionMask::mouseHover,
    };
}

void ImGuiMenu::UiRenderManagerImpl::finishBackwardPass( ViewportId viewport, const BasicUiRenderTask::BackwardPassParams& params )
{
    auto hoveredViewport = getViewerInstance().getHoveredViewportIdOrInvalid();

    if ( hoveredViewport != viewport )
    {
        if ( !hoveredViewport.valid() )
            consumedInteractions = {}; // No viewports are hovered, just zero this.

        // Otherwise we have some hovered viewport, but it's not this one, so we let that one viewport set `consumedInteractions`.
    }
    else if ( ImGui::GetIO().WantCaptureMouse )
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
