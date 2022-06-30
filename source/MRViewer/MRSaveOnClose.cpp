#include "MRSaveOnClose.h"
#include "MRMenu.h"
#include "MRFileDialog.h"
#include "MRProgressBar.h"
#include <MRMesh/MRHistoryStore.h>
#include <MRMesh/MRSerializer.h>
#include "ImGuiHelpers.h"
#include "MRPch/MRSpdlog.h"
#include <GLFW/glfw3.h>

namespace MR
{

void SaveOnClosePlugin::preDraw_()
{
    if ( !showCloseModal_ )
        return;

    auto& viewerRef = MR::Viewer::instanceRef();
    if ( !viewerRef.getGlobalHistoryStore() || !viewerRef.getGlobalHistoryStore()->isSceneModified() )
    {
        glfwSetWindowShouldClose( Viewer::instance()->window, true );
        shouldClose_ = true;
        return;
    }

    auto menuInstance = viewerRef.getMenuPluginAs<MR::Menu>();
    if ( !menuInstance )
        return;
    auto* context = menuInstance->getCurrentContext();
    if ( !context )
        return;
    ImGui::SetCurrentContext( context );

    ImGui::OpenPopup( "Application close" );
    ImGui::SetNextWindowSize( ImVec2( 300 * menuInstance->menu_scaling(), -1 ), ImGuiCond_Always );
    ImGui::BeginModalNoAnimation( "Application close", nullptr, ImGuiWindowFlags_NoResize );

    ImGui::Text( "Save your changes?" );

    float w = ImGui::GetContentRegionAvail().x;
    float p = ImGui::GetStyle().FramePadding.x;
    if ( ImGui::Button( "Save", ImVec2( ( w - p ) / 3.f, 0 ) ) )
    {
        auto savePath = SceneRoot::getScenePath();
        if ( savePath.empty() )
            savePath = saveFileDialog( { {}, {},SceneFileFilters } );
        
        showCloseModal_ = false;
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
    if ( ImGui::IsItemHovered() )
        ImGui::SetTooltip( "Save the current scene and close the application" );

    ImGui::SameLine( 0, p );
    if ( ImGui::Button( "Close", ImVec2( ( w - p ) / 3.f, 0 ) ) )
    {
        glfwSetWindowShouldClose( Viewer::instance()->window, true );
        shouldClose_ = true;
        showCloseModal_ = false;
    }
    if ( ImGui::IsItemHovered() )
        ImGui::SetTooltip( "Close the application without saving" );

    ImGui::SameLine( 0, p );
    if ( ImGui::Button( "Cancel", ImVec2( ( w - p ) / 3.f, 0 ) ) )
    {
        showCloseModal_ = false;
    }
    if ( ImGui::IsItemHovered() )
        ImGui::SetTooltip( "Do not close the application" );

    if ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) )
        showCloseModal_ = false;

    ImGui::EndPopup();
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
    showCloseModal_ = true;
    return true;
}

MRVIEWER_PLUGIN_REGISTRATION( SaveOnClosePlugin )

} //namespace MR
