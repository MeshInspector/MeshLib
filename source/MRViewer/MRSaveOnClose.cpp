#include "MRSaveOnClose.h"
#include "ImGuiMenu.h"
#include "MRFileDialog.h"
#include "MRProgressBar.h"
#include "MRRibbonButtonDrawer.h"
#include "MRRibbonFontManager.h"
#include "MRViewer.h"
#include "MRHistoryStore.h"
#include "MRRibbonConstants.h"
#include "MRUIStyle.h"
#include "MRSceneCache.h"
#include "MRShowModal.h"
#include <MRMesh/MRIOFormatsRegistry.h>
#include <MRMesh/MRSerializer.h>
#include <MRMesh/MRObjectSave.h>
#include <MRMesh/MRSceneRoot.h>
#include <MRMesh/MRVisualObject.h>
#include "ImGuiHelpers.h"
#include "MRPch/MRSpdlog.h"
#include "MRUISaveChangesPopup.h"
#include <imgui_internal.h>
#include <GLFW/glfw3.h>

namespace MR
{

void SaveOnClosePlugin::preDraw_()
{
    if ( !initialized_ )
        return;

    float scaling = 1.0f;    
    if ( auto menuInstance = getViewerInstance().getMenuPlugin() )
        scaling = menuInstance->menu_scaling();

    if ( showCloseModal_ )
    {
        auto* modal = ImGui::GetTopMostPopupModal();
        auto& viewerRef = getViewerInstance();
        bool noModalWasPresent = activeModalHighlightTimer_ == 2.0f;
        if ( !modal && ( !viewerRef.getGlobalHistoryStore() || !viewerRef.getGlobalHistoryStore()->isSceneModified() ) && noModalWasPresent )
        {
            glfwSetWindowShouldClose( Viewer::instance()->window, true );
            shouldClose_ = true;
            showCloseModal_ = false;
        }
        if ( modal && activeModalHighlightTimer_ > 0.0f )
        {
            if ( int( activeModalHighlightTimer_ / 0.2f ) % 2 == 1 )
                ImGui::GetForegroundDrawList()->AddRect(
                    ImVec2( modal->Pos.x - 2.0f * scaling, modal->Pos.y - 2.0f * scaling ), 
                    ImVec2( modal->Pos.x + modal->Size.x + 2.0f * scaling, modal->Pos.y + modal->Size.y + 2.0f * scaling ),
                    Color::yellow().getUInt32(), 0.0f, 0, 2.0f * scaling );
            getViewerInstance().incrementForceRedrawFrames();
            activeModalHighlightTimer_ -= ImGui::GetIO().DeltaTime;
            if ( activeModalHighlightTimer_ < 0.0f )
                showCloseModal_ = false;
        }
        else if ( noModalWasPresent )
        {
            ImGui::OpenPopup( "Application Close##modal" );
            showCloseModal_ = false;
        }
        else
        {
            showCloseModal_ = false;
        }
    }

    UI::SaveChangesPopupSettings settings;
    settings.scaling = scaling;
    settings.header = "Application Close";
    settings.saveTooltip = "Save the current scene and close the application";
    settings.dontSaveTooltip = "Close the application without saving";
    settings.cancelTooltip = "Do not close the application";
    settings.onOk = [this] ()
    {
        glfwSetWindowShouldClose( Viewer::instance()->window, true );
        shouldClose_ = true;
    };
    UI::saveChangesPopup( 
        "Application Close##modal",
        settings );
}

void SaveOnClosePlugin::init( Viewer* _viewer )
{
    if ( !_viewer )
        return;
    if ( !bool( _viewer->getMenuPlugin() ) )
        return;
    viewer = _viewer;
    connect( viewer );
    initialized_ = true;
}

void SaveOnClosePlugin::shutdown()
{
    if ( !initialized_ )
        return;
    disconnect();
    viewer = nullptr;
}

bool SaveOnClosePlugin::interruptClose_()
{
    if ( !initialized_ )
        return false;
    if ( shouldClose_ )
        return false;
    activeModalHighlightTimer_ = 2.0f;
    showCloseModal_ = true;
    return true;
}

MRVIEWER_PLUGIN_REGISTRATION( SaveOnClosePlugin )

} //namespace MR
