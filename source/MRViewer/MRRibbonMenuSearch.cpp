#include "MRRibbonMenuSearch.h"
#include "ImGuiHelpers.h"
#include "MRColorTheme.h"
#include "MRRibbonFontManager.h"
#include "MRRibbonConstants.h"
#include "MRRibbonButtonDrawer.h"
#include <imgui_internal.h>
#include "MRViewerInstance.h"
#include "MRViewer/MRViewer.h"
#include "MRRibbonMenu.h"

namespace MR
{

constexpr float SearchSize = 300.0f;

void RibbonMenuSearch::pushRecentItem( const std::shared_ptr<RibbonMenuItem>& item )
{
    if ( !item )
        return;
    auto it = std::find_if( recentItems_.begin(), recentItems_.end(), [&] ( const auto& other )
    {
        return other.item->item == item;
    } );
    if ( it != recentItems_.end() )
    {
        std::rotate( recentItems_.begin(), it, it + 1 );
        return;
    }

    auto sIt = RibbonSchemaHolder::schema().items.find( item->name() );
    if ( sIt == RibbonSchemaHolder::schema().items.end() )
    {
        assert( false );
        return;
    }

    RibbonSchemaHolder::SearchResult res;
    res.item = &sIt->second;
    if ( recentItems_.size() < 10 )
        recentItems_.insert( recentItems_.begin(), std::move( res ) );
    else
    {
        std::rotate( recentItems_.begin(), recentItems_.end() - 1, recentItems_.end() );
        recentItems_.front() = std::move( res );
    }
}

void RibbonMenuSearch::draw( const Parameters& params )
{
    if ( !ImGui::IsPopupOpen( windowName() ) )
    {
        if ( popupWasOpen_ )
        {
            searchLine_.clear();
            searchResult_.clear();
            hightlightedSearchItem_ = -1;
            popupWasOpen_ = false;
        }
        return;
    }

    popupWasOpen_ = true;
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
    if ( ImGui::Begin( windowName(), NULL, window_flags ) )
    {
        // Search line
        bool appearing = ImGui::IsWindowAppearing();

        const float minSearchSize = SearchSize * params.scaling;
        if ( isSmallUI() )
        {
            if ( appearing )
                ImGui::SetKeyboardFocusHere();

            ImGui::SetNextItemWidth( minSearchSize );
            if ( ImGui::InputText( "##SearchLine", searchLine_ ) )
                searchResult_ = RibbonSchemaHolder::search( searchLine_ );
        }

        const auto& resultsList = searchLine_.empty() ? recentItems_ : searchResult_;

        if ( resultsList.empty() )
            hightlightedSearchItem_ = -1;

        if ( !appearing )
        {
            if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
                ImGui::CloseCurrentPopup();
            else if ( ImGui::IsKeyPressed( ImGuiKey_DownArrow ) && hightlightedSearchItem_ + 1 < resultsList.size() )
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
//        int uniqueBtnCounter = 0;
        for ( int i = 0; i < resultsList.size(); ++i )
        {
            const auto& foundItem = resultsList[i];
            if ( !foundItem.item )
                continue;
            //auto pos = ImGui::GetCursorPos();
            auto width = params.btnDrawer.calcItemWidth( *foundItem.item, DrawButtonParams::SizeType::SmallText );
            DrawButtonParams dbParams;
            dbParams.sizeType = DrawButtonParams::SizeType::SmallText;
            dbParams.iconSize = cSmallIconSize;
            dbParams.itemSize.y = ySize;
            dbParams.itemSize.x = width.baseWidth + width.additionalWidth + 2.0f * cRibbonButtonWindowPaddingX * params.scaling;
            dbParams.forceHovered = hightlightedSearchItem_ == i;
            dbParams.forcePressed = dbParams.forceHovered &&
                ( ImGui::IsKeyPressed( ImGuiKey_Enter ) || ImGui::IsKeyPressed( ImGuiKey_KeypadEnter ) );
            //ImGui::SetCursorPosY( pos.y );
            params.btnDrawer.drawButtonItem( *foundItem.item, dbParams );
        }
        ImGui::PopStyleVar( 1 );
        ImGui::PopStyleColor( 3 );
        ImGui::PopFont();
        ImGui::EndPopup();
    }
}

void RibbonMenuSearch::drawMenuUI()
{
    auto ribbonMenu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
    const auto scaling = ribbonMenu ? ribbonMenu->menu_scaling() : 1.f;

    if ( isSmallUI() )
    {
        bool popupOpened = ImGui::IsPopupOpen( windowName() );

        ImFont* font = nullptr;
        if ( ribbonMenu )
            font = ribbonMenu->getFontManager().getFontByType( RibbonFontManager::FontType::Icons );
        if ( font )
            font->Scale = 0.7f;

        ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding * scaling );
        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
        if ( popupOpened )
            ImGui::PushStyleColor( ImGuiCol_Button, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );
        else
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );
        ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText ).getUInt32() );

        float btnSize = scaling * cTopPanelAditionalButtonSize;
        if ( font )
            ImGui::PushFont( font );
        auto pressed = ImGui::Button( "\xef\x80\x82", ImVec2( btnSize, btnSize ) );
        if ( font )
        {
            ImGui::PopFont();
            font->Scale = 1.0f;
        }

        ImGui::PopStyleColor( 4 );
        ImGui::PopStyleVar( 2 );

        // manage search popup
        if ( pressed && !popupOpened )
            ImGui::OpenPopup( windowName() );
    }
    else
    {
        ImGui::SetNextItemWidth( 285.f * scaling );
        const bool wasEmpty = searchLine_.empty();
        if ( ImGui::InputText( "##SearchLine", searchLine_ ) )
            searchResult_ = RibbonSchemaHolder::search( searchLine_ );

        if ( wasEmpty && !searchLine_.empty() )
            ImGui::OpenPopup( windowName() );
    }

}

bool RibbonMenuSearch::isSmallUI()
{
    auto ribbonMenu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
    const auto scaling = ribbonMenu ? ribbonMenu->menu_scaling() : 1.f;
    return getViewerInstance().framebufferSize.x < 1200 * scaling;
}

float RibbonMenuSearch::getWidthMenuUI()
{
    return isSmallUI() ? 40.f : 300.f;
}

}
