#include "MRRibbonMenuSearch.h"
#include "ImGuiHelpers.h"
#include "MRColorTheme.h"
#include "MRRibbonFontManager.h"
#include "MRRibbonConstants.h"
#include "MRRibbonButtonDrawer.h"
#include <imgui_internal.h>
#include "MRViewer/MRUITestEngine.h"
#include "MRViewerInstance.h"
#include "MRViewer/MRViewer.h"
#include "MRRibbonMenu.h"
#include "MRUIStyle.h"

namespace MR
{

constexpr float cSearchSize = 250.f;

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

    const auto& schema = RibbonSchemaHolder::schema();
    auto sIt = schema.items.find( item->name() );
    if ( sIt == schema.items.end() )
    {
        // no need to assert here, we could fall int this function from LambdaRibbonItem that is not present in scheme
        //assert( false );
        return;
    }

    RibbonSchemaHolder::SearchResult res;
    res.item = &sIt->second;
    res.tabIndex = RibbonSchemaHolder::findItemTab( item );

    if ( recentItems_.size() < 10 )
        recentItems_.insert( recentItems_.begin(), std::move( res ) );
    else
    {
        std::rotate( recentItems_.begin(), recentItems_.end() - 1, recentItems_.end() );
        recentItems_.front() = std::move( res );
    }
}

void RibbonMenuSearch::drawWindow_( const Parameters& params )
{
    // copy `recentItems_` because it can be rotated in THIS frame which is not good
    std::vector<RibbonSchemaHolder::SearchResult> recentItemsCpy;
    if ( searchLine_.empty() )
        recentItemsCpy = recentItems_;
    const auto& resultsList = searchLine_.empty() ? recentItemsCpy : searchResult_;

    if ( !isSmallUI_ && resultsList.empty() )
        return;

    UI::TestEngine::pushTree( "RibbonSearchPopup" );
    MR_FINALLY{ UI::TestEngine::popTree(); };

    const float screenWidth = float( getViewerInstance().framebufferSize.x );
    const float windowPaddingX = ImGui::GetStyle().WindowPadding.x;
    ImVec2 pos;
    pos.x = std::max( screenWidth - ( 70.f + cSearchSize + 16.f ) * params.scaling - windowPaddingX, 0.f );
    pos.y = ( cTabYOffset + cTabHeight ) * params.scaling;
    ImGui::SetNextWindowPos( pos );
    ImGui::SetNextWindowSize( ImVec2( ( cSearchSize + 20 ) * params.scaling, -1 ) );

    ImGui::SetNextFrameWantCaptureKeyboard( true );
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoFocusOnAppearing;
    if ( ImGui::Begin( windowName(), NULL, window_flags ) )
    {
        if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
            deactivateSearch_();
#ifndef NDEBUG
        if ( ImGui::IsKeyPressed( ImGuiKey_F11 ) )
            showResultWeight_ = !showResultWeight_;
#endif

        const float minSearchSize = cSearchSize * params.scaling;
        if ( isSmallUI_ )
        {
            if ( !isSmallUILast_ || ImGui::IsWindowAppearing() || setInputFocus_ )
            {
                ImGui::SetKeyboardFocusHere();
                setInputFocus_ = false;
            }
            ImGui::SetNextItemWidth( minSearchSize );
            if ( UI::inputText( "##SearchLine", searchLine_ ) )
                updateSearchResult_();
            if ( !ImGui::IsWindowAppearing() &&
                !( ImGui::IsWindowFocused() || ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) ) )
                deactivateSearch_();

            if ( ImGui::IsItemDeactivated() )
            {
                if ( ImGui::IsKeyPressed( ImGuiKey_Enter ) || ImGui::IsKeyPressed( ImGuiKey_KeypadEnter ) )
                    setInputFocus_ = true;
            }
        }
        else
        {
            if ( !mainInputFocused_ && !isSmallUILast_ &&
                !( ImGui::IsWindowFocused() || ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) ) )
                deactivateSearch_();
        }

        bool scroll = false;
        if ( !resultsList.empty() )
        {
            if ( ImGui::IsKeyPressed( ImGuiKey_DownArrow ) && hightlightedSearchItem_ + 1 < resultsList.size() )
            {
                hightlightedSearchItem_++;
                scroll = true;
            }
            else if ( ImGui::IsKeyPressed( ImGuiKey_UpArrow ) && hightlightedSearchItem_ > 0 )
            {
                hightlightedSearchItem_--;
                scroll = true;
            }
        }

        ImGui::PushFont( RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Small ) );
        auto ySize = ( cSmallIconSize + 2 * cRibbonButtonWindowPaddingY ) * params.scaling;
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered,
                               ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive,
                               ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActive ).getUInt32() );
        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
        bool openChild = false;
        if ( !resultsList.empty() )
        {
            openChild = true;
            const int itemCount = std::min( int( resultsList.size() ) + ( !searchLine_.empty() && captionCount_ >= 0 ? 1 : 0 ), 15 );
            float height = ySize * itemCount + ImGui::GetStyle().ItemSpacing.y * ( itemCount - 1 );
            height = std::min( height, getViewerInstance().framebufferSize.y - pos.y - ImGui::GetCursorPosY() - ImGui::GetStyle().WindowPadding.y );
            ImGui::BeginChild( "Search result list", ImVec2( -1, height ) );
        }
        for ( int i = 0; i < resultsList.size(); ++i )
        {
            const auto& foundItem = resultsList[i];
            if ( captionCount_ == i && !searchLine_.empty() )
            {
                if ( ImGui::BeginTable( "##Extended Search separator", 2, ImGuiTableFlags_SizingFixedFit) )
                {
                    ImGui::TableNextColumn();
                    ImGui::Text( "Extended Search" );
                    ImGui::TableNextColumn();
                    auto width = ImGui::GetWindowWidth();
                    ImGui::SetCursorPos( { width - ImGui::GetStyle().WindowPadding.x, ImGui::GetCursorPosY() + std::round( ImGui::GetTextLineHeight() * 0.5f ) } );
                    ImGui::Separator();
                    ImGui::EndTable();
                }
            }
            if ( scroll && hightlightedSearchItem_ == i )
                ImGui::SetScrollHereY();
            auto width = params.btnDrawer.calcItemWidth( *foundItem.item, DrawButtonParams::SizeType::SmallText );
            DrawButtonParams dbParams;
            dbParams.sizeType = DrawButtonParams::SizeType::SmallText;
            dbParams.iconSize = cSmallIconSize;
            dbParams.itemSize.y = ySize;
            dbParams.itemSize.x = width.baseWidth + width.additionalWidth + 2.0f * cRibbonButtonWindowPaddingX * params.scaling;
            dbParams.forceHovered = hightlightedSearchItem_ == i;
            dbParams.forcePressed = dbParams.forceHovered &&
                ( ImGui::IsKeyPressed( ImGuiKey_Enter ) || ImGui::IsKeyPressed( ImGuiKey_KeypadEnter ) );
            const bool pluginActive = foundItem.item->item->isActive();
            const float tabBtnWidth = 76 * params.scaling;
            const float tabBtnPadding = 8 * params.scaling;
            if ( foundItem.tabIndex != -1 )
            {
                auto storePos = ImGui::GetCursorPos();
                auto numColors = params.btnDrawer.pushRibbonButtonColors( true, false, false, dbParams.rootType );

                auto name = "##" + RibbonSchemaHolder::schema().tabsOrder[foundItem.tabIndex].name + "##" + foundItem.item->item->name();

                if ( ImGui::Button( name.c_str(), ImVec2( tabBtnWidth, dbParams.itemSize.y ) ) )
                    params.changeTabFunc( foundItem.tabIndex );

                auto textSize = ImGui::CalcTextSize( RibbonSchemaHolder::schema().tabsOrder[foundItem.tabIndex].name.c_str() );

                ImGui::SetCursorPosX( storePos.x + ( tabBtnWidth - textSize.x ) * 0.5f );
                ImGui::SetCursorPosY( storePos.y + ( dbParams.itemSize.y - textSize.y ) * 0.5f );
                ImGui::Text( "%s", RibbonSchemaHolder::schema().tabsOrder[foundItem.tabIndex].name.c_str() );

                ImGui::SetCursorPosX( storePos.x + tabBtnWidth + 0.5f * tabBtnPadding - params.scaling );
                ImGui::SetCursorPosY( storePos.y + ( dbParams.itemSize.y - textSize.y ) * 0.5f );
                ImGui::Text( ":" );

                if ( numColors > 0 )
                    ImGui::PopStyleColor( numColors );
                ImGui::SetCursorPosY( storePos.y );
            }
            ImGui::SetCursorPosX( tabBtnWidth + tabBtnPadding );
            bool activated = false;
            dbParams.isPressed = &activated;
            params.btnDrawer.drawButtonItem( *foundItem.item, dbParams );
            if ( activated )
                pushRecentItem( foundItem.item->item );
            if ( foundItem.item->item->isActive() != pluginActive )
            {
                onToolActivateSignal( foundItem.item->item );
                deactivateSearch_();
            }
#ifndef NDEBUG
            if ( showResultWeight_ && !searchLine_.empty() )
            {
                const auto& weights = searchResultWeight_[i];
                ImGui::SameLine();
                ImGui::Text( "(?)" );
                if ( ImGui::IsItemHovered() )
                    ImGui::SetTooltip( "caption = %.3f\ncaption order = %.3f\ntooltip = %.3f\ntooltip order = %.3f",
                        weights.captionWeight, weights.captionOrderWeight,
                        weights.tooltipWeight, weights.tooltipOrderWeight );
            }
#endif
        }
        if ( openChild )
            ImGui::EndChild();
        ImGui::PopStyleVar( 1 );
        ImGui::PopStyleColor( 3 );
        ImGui::PopFont();
        ImGui::End();
    }
}

void RibbonMenuSearch::deactivateSearch_()
{
    active_ = false;
    searchLine_.clear();
    searchResult_.clear();
    searchResultWeight_.clear();
    setInputFocus_ = false;
    hightlightedSearchItem_ = 0;
}

void RibbonMenuSearch::drawMenuUI( const Parameters& params )
{
    UI::TestEngine::pushTree( "RibbonSearch" );
    if ( isSmallUI_ )
    {
        if ( smallSearchButton_( params ) )
        {
            if ( blockSearchBtn_ )
                blockSearchBtn_ = false;
            else
                active_ = true;
        }
        if ( ImGui::IsItemActivated() && active_ )
            blockSearchBtn_ = true;
    }
    else
    {
        if ( ( isSmallUILast_ && active_ ) || setInputFocus_ )
        {
            ImGui::SetKeyboardFocusHere();
            setInputFocus_ = false;
        }
        if ( searchInputText_( "##SearchLine", searchLine_, params ) )
            updateSearchResult_();
        if ( mainInputFocused_ && !ImGui::IsItemFocused() )
        {
            if ( ( !searchLine_.empty() && searchResult_.empty() ) || ( searchLine_.empty() && recentItems_.empty() ) )
                deactivateSearch_();
        }
        mainInputFocused_ = ImGui::IsItemFocused();
        if ( ImGui::IsItemActivated() )
            active_ = true;
        if ( ImGui::IsItemDeactivated() )
        {
            if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
                deactivateSearch_();
            if ( ImGui::IsKeyPressed( ImGuiKey_Enter ) || ImGui::IsKeyPressed( ImGuiKey_KeypadEnter ) )
                setInputFocus_ = true;
        }
    }

    if ( !prevFrameActive_ && active_ )
        onFocusSignal();

    if ( active_ )
        drawWindow_( params );

    UI::TestEngine::popTree();

    prevFrameActive_ = active_;
    isSmallUILast_ = isSmallUI_;
}

float RibbonMenuSearch::getWidthMenuUI() const
{
    return isSmallUI_ ? 40.f : cSearchSize + 16.f;
}

float RibbonMenuSearch::getSearchStringWidth() const
{
    return cSearchSize + 16.f;
}

void RibbonMenuSearch::activate()
{
    active_ = true;
    if ( !isSmallUI_ )
        setInputFocus_ = true;
}

bool RibbonMenuSearch::smallSearchButton_( const Parameters& params )
{
    ImFont* font = params.fontManager.getFontByType( RibbonFontManager::FontType::Icons );
    if ( font )
        font->Scale = 0.7f;

    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding * params.scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    if ( active_ )
        ImGui::PushStyleColor( ImGuiCol_Button, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );
    else
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );
    ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText ).getUInt32() );

    float btnSize = params.scaling * cTopPanelAditionalButtonSize;
    if ( font )
        ImGui::PushFont( font );
    bool pressed = UI::buttonEx( "\xef\x80\x82", ImVec2( btnSize, btnSize ), { .forceImGuiBackground = true, .testEngineName = "ActivateSearchBtn" } );
    if ( font )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    ImGui::PopStyleColor( 4 );
    ImGui::PopStyleVar( 2 );

    return pressed;
}

bool RibbonMenuSearch::searchInputText_( const char* label, std::string& str, const RibbonMenuSearch::Parameters& params )
{
    ImGui::PushID( "searchInputText" );
    const ImVec2 cursorPos = ImGui::GetCursorPos();

    const auto& style = ImGui::GetStyle();

    const float inputHeight = ImGui::GetTextLineHeight() + style.FramePadding.y * 2.f;
    const auto drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled( cursorPos, ImVec2( cursorPos.x + cSearchSize * params.scaling, cursorPos.y + inputHeight ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TopPanelSearchBackground ).getUInt32(), style.FrameRounding );
    drawList->AddRect( cursorPos, ImVec2( cursorPos.x + cSearchSize * params.scaling, cursorPos.y + inputHeight ),
        ImGui::GetColorU32( ImGuiCol_Border ), style.FrameRounding );

    int colorNum = 0;
    if ( !active_ )
    {
        ImGui::PushStyleColor( ImGuiCol_Text, Color::gray().getUInt32() );
        ++colorNum;
    }
    ImFont* font = params.fontManager.getFontByType( RibbonFontManager::FontType::Icons );
    if ( font )
        font->Scale = 0.7f;
    if ( font )
        ImGui::PushFont( font );
    const float inputWidth = cSearchSize * params.scaling - style.FramePadding.x - style.ItemSpacing.x - ImGui::CalcTextSize( "\xef\x80\x82" ).x;
    ImGui::SetCursorPos( ImVec2( cursorPos.x + inputWidth + style.ItemSpacing.x, cursorPos.y + style.FramePadding.y ) );
    ImGui::Text( "%s", "\xef\x80\x82" );
    if ( font )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }
    if ( colorNum )
        ImGui::PopStyleColor( colorNum );

    if ( ImGui::IsItemClicked( ImGuiMouseButton_Left ) )
        activate();

    ImGui::SetCursorPos( cursorPos );
    ImGui::SetNextItemWidth( inputWidth );
    ImGui::PushStyleColor( ImGuiCol_FrameBg, Color::transparent().getUInt32() );
    ImGui::PushStyleColor( ImGuiCol_Border, Color::transparent().getUInt32() );
    const bool res = UI::inputText( label, str );
    ImGui::PopStyleColor( 2 );

    ImGui::PopID();

    return res;
}

void RibbonMenuSearch::updateSearchResult_()
{
    searchResult_ = RibbonSchemaHolder::search( searchLine_, { &captionCount_, &searchResultWeight_, requirementsFunc_ } );
    hightlightedSearchItem_ = 0;
}

}
