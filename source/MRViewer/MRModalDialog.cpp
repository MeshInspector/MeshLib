#include "MRModalDialog.h"

#include "ImGuiHelpers.h"
#include "MRRibbonConstants.h"
#include "MRRibbonFontManager.h"
#include "MRUIStyle.h"

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

    const ImVec2 windowPadding { cModalWindowPaddingX * menuScaling, cModalWindowPaddingY * menuScaling };
    const ImVec2 itemSpacing { 2.0f * cDefaultItemSpacing * menuScaling, 3.0f * cDefaultItemSpacing * menuScaling };
    const ImVec2 itemInnerSpacing { 2.0f * cDefaultInnerSpacing * menuScaling, cDefaultInnerSpacing * menuScaling };
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, windowPadding );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, itemInnerSpacing );

    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
    if ( !ImGui::BeginModalNoAnimation( label_.c_str(), nullptr, flags ) )
    {
        ImGui::PopStyleVar(); // ImGuiStyleVar_ItemSpacing
        ImGui::PopStyleVar(); // ImGuiStyleVar_WindowPadding
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
        if ( textWidth < windowSize.x )
        {
            ImGui::SetCursorPosX( ( windowSize.x - textWidth ) * 0.5f );
            ImGui::Text( "%s", text.c_str() );
        }
        else
        {
            ImGui::TextWrapped( "%s", text.c_str() );
        }
    }

    return true;
}

void ModalDialog::endPopup( float )
{
    if ( auto* dontShowAgain = settings_.dontShowAgain )
    {
        constexpr const auto* cDontShowAgainText = "Don't show the dialog again";
        const auto checkboxWidth = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize( cDontShowAgainText ).x;
        ImGui::SetCursorPosX( ( windowWidth() - checkboxWidth ) * 0.5f );
        UI::checkbox( cDontShowAgainText, dontShowAgain );
    }

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

    ImGui::PopStyleVar(); // ImGuiStyleVar_ItemInnerSpacing
    ImGui::PopStyleVar(); // ImGuiStyleVar_ItemSpacing
    ImGui::PopStyleVar(); // ImGuiStyleVar_WindowPadding
}

float ModalDialog::windowWidth()
{
    return ImGui::GetCurrentWindow()->Size.x;
}

} // namespace MR
