#ifdef __EMSCRIPTEN__
#include "MRWasmUnavailablePlugin.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRUIStyle.h"
#include "MRMesh/MRSystem.h"

namespace MR
{

void WasmUnavailablePlugin::drawDialog( float scaling, ImGuiContext* )
{
    auto menuInstance = getViewerInstance().getMenuPlugin();
    if ( !menuInstance )
        return;

    if ( openPopup_ )
    {
        ImGui::OpenPopup( "##WasmBlocked" );
        openPopup_ = false;
    }

    const ImVec2 windowSize{ cModalWindowWidth * scaling * 1.5f, -1 };
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );

    
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 2.0f * cDefaultItemSpacing * scaling, cDefaultItemSpacing * scaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * scaling, cModalWindowPaddingY * scaling } );
    if ( ImGui::BeginModalNoAnimation( "##WasmBlocked", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        if ( ImGui::ModalBigTitle( "The tool is not supported by the browser version", scaling ) )
        {
            ImGui::CloseCurrentPopup();
            dialogIsOpen_ = false;
        }

        auto text = "We are sorry, this feature is not implemented in web version. Please install desktop version.";
        const float textWidth = ImGui::CalcTextSize( text ).x;
        if ( textWidth < windowSize.x )
        {
            ImGui::SetCursorPosX( ( windowSize.x - textWidth ) * 0.5f );
            ImGui::Text( "%s", text );
        }
        else
        {
            ImGui::TextWrapped( "%s", text );
        }
        ImGui::NewLine();
        const auto style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * scaling } );

        const Vector2f btnSize{ ImGui::CalcTextSize( "Download" ).x + 4.0f * cGradientButtonFramePadding * scaling, 0 };
        ImGui::SetCursorPosX( ( windowSize.x - btnSize.x ) * 0.5f );
        if ( UI::button( "Download", btnSize ) )
        {
            OpenLink( "https://meshinspector.com/download" );
            ImGui::CloseCurrentPopup();
            dialogIsOpen_ = false;
        }
        UI::setTooltipIfHovered( "Open page with latest releases.", scaling );

        if ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) )
        {
            ImGui::CloseCurrentPopup();
            dialogIsOpen_ = false;
        }
        ImGui::PopStyleVar();
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar( 2 );
}

}
#endif