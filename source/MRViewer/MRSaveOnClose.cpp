#include "MRSaveOnClose.h"
#include "MRMenu.h"
#include "MRFileDialog.h"
#include "MRProgressBar.h"
#include "MRRibbonButtonDrawer.h"
#include <MRMesh/MRHistoryStore.h>
#include <MRMesh/MRSerializer.h>
#include "ImGuiHelpers.h"
#include "MRPch/MRSpdlog.h"
#include "MRRibbonConstants.h"
#include <imgui_internal.h>
#include <GLFW/glfw3.h>

namespace MR
{

void SaveOnClosePlugin::preDraw_()
{
    float scaling = 1.0f;    
    if ( auto menuInstance = getViewerInstance().getMenuPlugin() )
        scaling = menuInstance->menu_scaling();

    if ( showCloseModal_ )
    {
        auto* modal = ImGui::GetTopMostPopupModal();
        auto& viewerRef = getViewerInstance();
        if ( !modal && ( !viewerRef.getGlobalHistoryStore() || !viewerRef.getGlobalHistoryStore()->isSceneModified() ) )
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
        else
        {
            ImGui::OpenPopup( "Application close##modal" );
            showCloseModal_ = false;
        }
    }
	ImGui::SetNextWindowSize( ImVec2( 300 * scaling, -1 ), ImGuiCond_Always );
    if ( ImGui::BeginModalNoAnimation( "Application close##modal", nullptr, ImGuiWindowFlags_NoResize ) )
    {

        ImGui::Text( "Save your changes?" );

		float p = ImGui::GetStyle().FramePadding.x;
        const float btnHeight = ImGui::CalcTextSize( "SDC" ).y + cGradientButtonFramePadding * scaling;
        const ImVec2 btnSize = ImVec2( ( ImGui::GetContentRegionAvail().x - p * 2 ) / 3.f, btnHeight );
        if ( RibbonButtonDrawer::GradientButton( "Save", btnSize ) )
        {
            auto savePath = SceneRoot::getScenePath();
            if ( savePath.empty() )
                savePath = saveFileDialog( { {}, {},SceneFileFilters } );

            ProgressBar::orderWithMainThreadPostProcessing( "Saving scene", [savePath, &root = SceneRoot::get(), viewer = Viewer::instance()]()->std::function<void()>
            {
                auto res = serializeObjectTree( root, savePath, ProgressBar::callBackSetProgress );
                if ( !res.has_value() )
                    spdlog::error( res.error() );

                return[savePath, viewer, success = res.has_value()]()
                {
                    if ( success )
                    {
                        viewer->onSceneSaved( savePath );
                        glfwSetWindowShouldClose( Viewer::instance()->window, true );
                    }
                };
            } );
        }
        ImGui::SetTooltipIfHovered( "Save the current scene and close the application", scaling );

        ImGui::SameLine( 0, p );
        if ( RibbonButtonDrawer::GradientButton( "Don't Save", btnSize ) )
        {
            glfwSetWindowShouldClose( Viewer::instance()->window, true );
            shouldClose_ = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetTooltipIfHovered( "Close the application without saving", scaling );

        ImGui::SameLine( 0, p );
        if ( RibbonButtonDrawer::GradientButton( "Cancel", btnSize ) )
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetTooltipIfHovered( "Do not close the application", scaling );

        if ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) )
            ImGui::CloseCurrentPopup();

        ImGui::EndPopup();
	}
}

void SaveOnClosePlugin::init( Viewer* _viewer )
{
    if ( !_viewer )
        return;
    viewer = _viewer;
    connect( viewer );
}

void SaveOnClosePlugin::shutdown()
{
    disconnect();
    viewer = nullptr;
}

bool SaveOnClosePlugin::interruptClose_()
{
    if ( shouldClose_ )
        return false;
    activeModalHighlightTimer_ = 2.0f;
    showCloseModal_ = true;
    return true;
}

MRVIEWER_PLUGIN_REGISTRATION( SaveOnClosePlugin )

} //namespace MR
