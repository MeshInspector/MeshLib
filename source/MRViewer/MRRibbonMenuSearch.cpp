#include "MRRibbonMenuSearch.h"
#include "ImGuiHelpers.h"
#include "MRColorTheme.h"
#include "MRRibbonFontManager.h"
#include "MRRibbonConstants.h"
#include "MRRibbonButtonDrawer.h"
#include <imgui_internal.h>

namespace MR
{

void RibbonMenuSearch::draw( const Parameters& params )
{
    if ( ImGuiWindow* menuWindow = ImGui::FindWindowByName( windowName() ) )
        if ( menuWindow->WasActive )
        {
            ImRect frame;
            frame.Min = params.absMinPos;
            frame.Max = ImVec2( frame.Min.x + ImGui::GetFrameHeight(), frame.Min.y + ImGui::GetFrameHeight() );
            ImVec2 expectedSize = ImGui::CalcWindowNextAutoFitSize( menuWindow );
            menuWindow->AutoPosLastDirection = ImGuiDir_Down;
            ImRect rectOuter = ImGui::GetPopupAllowedExtentRect( menuWindow );
            ImVec2 pos = ImGui::FindBestWindowPosForPopupEx( frame.GetBL(), expectedSize, &menuWindow->AutoPosLastDirection, rectOuter, frame, ImGuiPopupPositionPolicy_ComboBox );
            ImGui::SetNextWindowPos( pos );
        }

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_Popup | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove;
    if ( ImGui::Begin( windowName(), NULL, window_flags) )
    {
        // Search line
        bool appearing = ImGui::IsWindowAppearing();
        if ( appearing )
        {
            searchLine_.clear();
            searchResult_.clear();
            hightlightedSearchItem_ = -1;
            ImGui::SetKeyboardFocusHere();
        }
        float minSearchSize = 300.0f * params.scaling;
        ImGui::SetNextItemWidth( minSearchSize );
        if ( ImGui::InputText( "##SearchLine", searchLine_ ) )
            searchResult_ = RibbonSchemaHolder::search( searchLine_ );

        if ( searchResult_.empty() )
            hightlightedSearchItem_ = -1;

        if ( !appearing )
        {
            if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
                ImGui::CloseCurrentPopup();
            else if ( ImGui::IsKeyPressed( ImGuiKey_DownArrow ) && hightlightedSearchItem_ + 1 < searchResult_.size() )
                hightlightedSearchItem_++;
            else if ( ImGui::IsKeyPressed( ImGuiKey_UpArrow ) && hightlightedSearchItem_ > 0 )
                hightlightedSearchItem_--;
        }

        ImGui::PushFont( RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Small ) );
        auto ySize = ( cSmallIconSize + 2 * cRibbonButtonWindowPaddingY ) * params.scaling;
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered,
                               ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive,
                               ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActive ).getUInt32() );
        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
        int uniqueBtnCounter = 0;
        for ( int i = 0; i < searchResult_.size(); ++i )
        {
            const auto& foundItem = searchResult_[i];
            if ( !foundItem.item )
                continue;
            auto pos = ImGui::GetCursorPos();
            if ( foundItem.tabIndex != -1 )
            {
                const auto& tabName = RibbonSchemaHolder::schema().tabsOrder[foundItem.tabIndex].name;
                auto label = "##SearchTabBtn" + tabName + std::to_string( ++uniqueBtnCounter );
                auto labelSize = ImGui::CalcTextSize( tabName.c_str() );
                if ( ImGui::Button( label.c_str(), ImVec2( labelSize.x + 2 * cRibbonButtonWindowPaddingX * params.scaling, ySize ) ) )
                {
                    if ( params.changeTabFunc )
                        params.changeTabFunc( foundItem.tabIndex );
                    ImGui::CloseCurrentPopup();
                }
                ImVec2 textPos = pos;
                textPos.x += cRibbonButtonWindowPaddingX * params.scaling;
                textPos.y += ( ySize - labelSize.y ) * 0.5f;
                ImGui::SetCursorPos( textPos );
                ImGui::Text( "%s", tabName.c_str() );
                ImGui::SameLine( 0.0f, cRibbonButtonWindowPaddingX * params.scaling + ImGui::GetStyle().ItemSpacing.x );
                ImGui::SetCursorPosX( minSearchSize * 0.3f );
                ImGui::Text( ">" );
                ImGui::SameLine( 0.0f, cRibbonButtonWindowPaddingX * params.scaling + ImGui::GetStyle().ItemSpacing.x );
            }
            auto width = params.btnDrawer.calcItemWidth( *foundItem.item, DrawButtonParams::SizeType::SmallText );
            DrawButtonParams dbParams;
            dbParams.sizeType = DrawButtonParams::SizeType::SmallText;
            dbParams.iconSize = cSmallIconSize;
            dbParams.itemSize.y = ySize;
            dbParams.itemSize.x = width.baseWidth + width.additionalWidth + 2.0f * cRibbonButtonWindowPaddingX * params.scaling;
            dbParams.forceHovered = hightlightedSearchItem_ == i;
            dbParams.forcePressed = dbParams.forceHovered &&
                ( ImGui::IsKeyPressed( ImGuiKey_Enter ) || ImGui::IsKeyPressed( ImGuiKey_KeypadEnter ) );
            ImGui::SetCursorPosY( pos.y );
            params.btnDrawer.drawButtonItem( *foundItem.item, dbParams );
        }
        ImGui::PopStyleVar( 1 );
        ImGui::PopStyleColor( 3 );
        ImGui::PopFont();
        ImGui::EndPopup();
    }
}

}
