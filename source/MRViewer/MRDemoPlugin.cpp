#include "MRDemoPlugin.h"
#include "MRMenu.h"
#include "MRMesh/MRUVSphere.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMesh.h"
#include "ImGuiHelpers.h"
#include <GLFW/glfw3.h>

namespace MR
{

void DemoPlugin::draw_()
{
    viewer->viewport().draw( *demoSphere_, demoSphere_->xf(), true );
    viewer->viewport().draw( *demoSphere_, demoSphere_->xf() );
}

void DemoPlugin::preDraw_()
{
    auto& viewerRef = MR::Viewer::instanceRef();
    auto menuInstance = viewerRef.getMenuPluginAs<MR::Menu>();
    if ( !menuInstance )
        return;
    auto* context = menuInstance->getCurrentContext();
    if ( !context )
        return;
    ImGui::SetCurrentContext( context );
    ImGui::SetNextWindowSize( ImVec2( 100, 100 ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowPos( ImVec2( 500, 500 ), ImGuiCond_FirstUseEver );
    ImGui::Begin( "Demo Plugin window", nullptr );
    ImGui::Text( "DEMO" );
    ImGui::End();

    if ( showCloseModal_ )
    {
        ImGui::OpenPopup( "Close##sureClose" );
        ImGui::SetNextWindowSize( ImVec2( 200 * menuInstance->menu_scaling(), -1 ), ImGuiCond_Always );
        ImGui::BeginModalNoAnimation( "Close##sureClose", nullptr, ImGuiWindowFlags_NoResize );

        ImGui::Text( "Are you sure?" );

        float w = ImGui::GetContentRegionAvail().x;
        float p = ImGui::GetStyle().FramePadding.x;
        if ( ImGui::Button( "Ok", ImVec2( ( w - p ) / 2.f, 0 ) ) )
        {
            glfwSetWindowShouldClose( Viewer::instance()->window, true );
            shouldClose_ = true;
            showCloseModal_ = false;
        }
        ImGui::SameLine( 0, p );
        if ( ImGui::Button( "Cancel", ImVec2( ( w - p ) / 2.f, 0 ) ) )
            showCloseModal_ = false;

        if ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) )
            showCloseModal_ = false;

        ImGui::EndPopup();
    }
}

void DemoPlugin::init( Viewer* _viewer )
{
    if ( !_viewer )
        return;
    viewer = _viewer;
    demoSphere_ = std::make_unique<ObjectMesh>();
    demoSphere_->setMesh( std::make_shared<Mesh>( makeUVSphere( 1.0f, 64, 64 ) ) );
    connect( viewer );
}

void DemoPlugin::shutdown()
{
    disconnect();
    viewer = nullptr;
    demoSphere_.reset();
}

bool DemoPlugin::interruptClose_()
{
    if ( shouldClose_ )
        return false;
    showCloseModal_ = true;
    return true;
}

DemoPlugin DemoPluginInstance;
}
