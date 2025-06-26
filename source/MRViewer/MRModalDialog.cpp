#include "MRModalDialog.h"

#include "ImGuiHelpers.h"
#include "MRRibbonConstants.h"
#include "MRRibbonFontManager.h"
#include "MRUIStyle.h"
#include "MRViewer.h"

namespace MR
{

ModalDialog::ModalDialog( std::string label, ModalDialogSettings settings )
    : label_( std::move( label ) )
    , settings_( std::move( settings ) )
{
}

bool ModalDialog::beginPopup( float menuScaling )
{
    const auto windowWidth = settings_.windowWidth > 0.f ? settings_.windowWidth : cModalWindowWidth * menuScaling;
    const ImVec2 windowSize { windowWidth, -1 };
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
    ImGui::SetNextWindowPos( Vector2f( getViewerInstance().framebufferSize ) * 0.5f, ImGuiCond_Always, ImVec2( 0.5f, 0.5f ) );

    setStyle_( menuScaling );

    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
    if ( !ImGui::BeginModalNoAnimation( label_.c_str(), nullptr, flags ) )
    {
        unsetStyle_();
        return false;
    }

    if ( const auto& headline = settings_.headline; !headline.empty() )
    {
        auto headerFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Headline );
        if ( headerFont )
            ImGui::PushFont( headerFont );

        const auto headlineWidth = ImGui::CalcTextSize( headline.c_str() ).x;
        ImGui::SetCursorPosX( ( windowSize.x - headlineWidth ) * 0.5f );
        ImGui::Text( "%s", headline.c_str() );

        if ( headerFont )
            ImGui::PopFont();
    }

    if ( settings_.closeButton )
    {
        const auto closeButtonWidth = StyleConsts::Modal::exitBtnSize * menuScaling;
        ImGui::SameLine( ImGui::GetWindowContentRegionMax().x - closeButtonWidth );
        if ( ImGui::ModalExitButton( menuScaling ) )
        {
            ImGui::CloseCurrentPopup();
            if ( settings_.onWindowClose )
                settings_.onWindowClose();
        }
    }

    if ( const auto& text = settings_.text; !text.empty() )
    {
        const auto textWidth = ImGui::CalcTextSize( text.c_str() ).x;
        if ( textWidth < ( windowSize.x - cModalWindowPaddingX * menuScaling * 2.f ) )
        {
            ImGui::SetCursorPosX( ( windowSize.x - textWidth ) * 0.5f );
            ImGui::Text( "%s", text.c_str() );
        }
        else
        {
            ImGui::TextWrapped( "%s", text.c_str() );
        }
    }

    if ( auto* dontShowAgain = settings_.dontShowAgain )
    {
        constexpr const auto* cDontShowAgainText = "Do not show this message again";
        const auto checkboxWidth = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize( cDontShowAgainText ).x;
        ImGui::SetCursorPosX( ( windowWidth - checkboxWidth ) * 0.5f );
        auto color = ImGui::GetStyleColorVec4( ImGuiCol_Text );
        color.w = 0.5f;
        ImGui::PushStyleColor( ImGuiCol_Text, color );
        UI::checkbox( cDontShowAgainText, dontShowAgain );
        ImGui::PopStyleColor();
    }

    return true;
}

void ModalDialog::endPopup( float )
{
    if ( settings_.closeOnClickOutside )
    {
        const auto clicked = ImGui::IsMouseClicked( ImGuiMouseButton_Left );
        const auto insideDialog = ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow );
        if ( clicked && !insideDialog )
        {
            ImGui::CloseCurrentPopup();
            if ( settings_.onWindowClose )
                settings_.onWindowClose();
        }
    }

    ImGui::EndPopup();
    unsetStyle_();
}

float ModalDialog::windowWidth()
{
    return ImGui::GetCurrentWindow()->Size.x;
}

void ModalDialog::setStyle_( float menuScaling )
{
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding,    { cModalWindowPaddingX * menuScaling, cModalWindowPaddingY * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing,      { 2.0f * cDefaultItemSpacing * menuScaling, 3.0f * cDefaultItemSpacing * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, { 2.0f * cDefaultInnerSpacing * menuScaling, cDefaultInnerSpacing * menuScaling } );
}

void ModalDialog::unsetStyle_()
{
    ImGui::PopStyleVar( 3 );
}

} // namespace MR
