#ifdef __EMSCRIPTEN__
#include "MRWasmUnavailablePlugin.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRUIStyle.h"
#include "MRMesh/MRSystem.h"

namespace MR
{

void WasmUnavailablePlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuInstance = getViewerInstance().getMenuPlugin();
    if ( !menuInstance )
        return;
    const auto scaling = menuInstance->menu_scaling();

    if ( openPopup_ )
    {
        ImGui::OpenPopup( "Unavailable Tool##WasmBlocked" );
        openPopup_ = false;
    }

    const ImVec2 windowSize{ cModalWindowWidth * scaling, -1 };
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );

    
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 2.0f * cDefaultItemSpacing * scaling, 3.0f * cDefaultItemSpacing * scaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * scaling, cModalWindowPaddingY * scaling } );
    if ( ImGui::BeginModalNoAnimation( "Unavailable Tool##WasmBlocked", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        auto headerFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Headline );
        if ( headerFont )
            ImGui::PushFont( headerFont );

        const auto headerWidth = ImGui::CalcTextSize( "Unavailable Tool" ).x;
        ImGui::SetCursorPosX( ( windowSize.x - headerWidth ) * 0.5f );
        ImGui::Text( "Unavailable Tool" );

        if ( headerFont )
            ImGui::PopFont();

        auto text = "This tool is unavailable due to some browser\nlimitations, you can use it in desktop verison.";
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

        const auto style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * scaling } );

        const float p = ImGui::GetStyle().ItemSpacing.x;
        const Vector2f btnSize{ ( ImGui::GetContentRegionAvail().x - p ) / 2.f, 0 };
        if ( UI::button( "Download", btnSize ) )
        {
            OpenLink( "https://github.com/MeshInspector/MeshInspector/releases" );
            ImGui::CloseCurrentPopup();
            dialogIsOpen_ = false;
        }
        UI::setTooltipIfHovered( "Open page with latest releases.", scaling );
        ImGui::SameLine();
        if ( UI::buttonCommonSize( "Cancel", btnSize, ImGuiKey_Escape ) )
        {
            ImGui::CloseCurrentPopup();
            dialogIsOpen_ = false;
        }

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