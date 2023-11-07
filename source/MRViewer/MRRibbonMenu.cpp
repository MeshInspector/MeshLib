#include "MRRibbonMenu.h"
#include "MRProgressBar.h"
#include "MRColorTheme.h"
#include "MRAppendHistory.h"
#include "MRCommandLoop.h"
#include "MRRibbonIcons.h"
#include "MRRibbonConstants.h"
#include "ImGuiHelpers.h"
#include "MRImGuiImage.h"
#include "MRFileDialog.h"
#include "MRViewerSettingsManager.h"
#include "MRUIStyle.h"
#include <MRMesh/MRString.h>
#include <MRMesh/MRSystem.h>
#include <MRMesh/MRStringConvert.h>
#include <MRMesh/MRSerializer.h>
#include <MRMesh/MRObjectsAccess.h>
#include <MRMesh/MRChangeXfAction.h>
#include <MRMesh/MRObjectLabel.h>
#include <MRMesh/MRChangeSceneObjectsOrder.h>
#include <MRMesh/MRChangeSceneAction.h>
#include <MRMesh/MRChangeObjectFields.h>
#include <MRPch/MRJson.h>
#include <MRPch/MRSpdlog.h>
#include <MRPch/MRWasm.h>
#include <imgui_internal.h> // needed here to fix items dialogs windows positions
#include <misc/freetype/imgui_freetype.h> // for proper font loading
#include <regex>

#if defined(__APPLE__) && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-volatile"
#endif

#include <GLFW/glfw3.h>

#if defined(__APPLE__) && defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace MR
{

void changeSelection( bool selectNext, int mod )
{
    using namespace MR;
    const auto selectable = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selectable );
    const auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selectNext )
    {
        auto nextIt = std::find_if( selectable.rbegin(), selectable.rend(), [] ( const std::shared_ptr<Object>& obj )
        {
            return obj->isSelected();
        } );

        Object* next{ nullptr };
        if ( nextIt != selectable.rend() )
        {
            auto dist = int( std::distance( nextIt, selectable.rend() ) );
            if ( dist >= 0 && dist < selectable.size() )
                next = selectable[dist].get();
            if ( dist == selectable.size() )
                next = selectable.back().get();
        }

        if ( mod == 0 ) // uncomment if want multy select holding shift
            for ( const auto& data : selected )
                if ( data && data.get() != next )
                    data->select( false );
        if ( next )
            next->select( true );
    }
    else
    {
        auto prevIt = std::find_if( selectable.begin(), selectable.end(), [] ( const std::shared_ptr<Object>& obj )
        {
            return obj->isSelected();
        } );

        Object* prev{ nullptr };
        if ( prevIt != selectable.end() )
        {
            auto dist = int( std::distance( selectable.begin(), prevIt ) ) - 1;
            if ( dist >= 0 && dist < selectable.size() )
                prev = selectable[dist].get();
            if ( dist == -1 )
                prev = selectable.front().get();
        }

        if ( mod == 0 ) // uncomment if want multy select holding shift
            for ( const auto& data : selected )
                if ( data && data.get() != prev )
                    data->select( false );
        if ( prev )
            prev->select( true );
    }
}

void RibbonMenu::init( MR::Viewer* _viewer )
{
    ImGuiMenu::init( _viewer );
    // should init instance before load schema (as far as some font are used inside)
    fontManager_.initFontManagerInstance( &fontManager_ );
    readMenuItemsStructure_();

    RibbonIcons::load();

    callback_draw_viewer_window = [] ()
    {};

    // Draw additional windows
    callback_draw_custom_window = [&] ()
    {
        prevFrameObjectsCache_ = selectedObjectsCache_;
        selectedObjectsCache_ = getAllObjectsInTree<const Object>( &SceneRoot::get(), ObjectSelectivityType::Selected );

        drawTopPanel_();

        drawActiveBlockingDialog_();
        drawActiveNonBlockingDialogs_();

        toolbar_.drawToolbar();
        toolbar_.drawCustomize();
        drawRibbonSceneList_();
        drawRibbonViewportsLabels_();
        
        drawActiveList_();

        drawWelcomeWindow_();

        draw_helpers();
        drawVersionWindow_();
    };

    buttonDrawer_.setMenu( this );
    buttonDrawer_.setShortcutManager( getShortcutManager().get() );
    buttonDrawer_.setScaling( menu_scaling() );
    buttonDrawer_.setOnPressAction( [&] ( std::shared_ptr<RibbonMenuItem> item, bool available )
    {
        itemPressed_( item, available );
    } );
    buttonDrawer_.setGetterRequirements( [&] ( std::shared_ptr<RibbonMenuItem> item )
    {
        return getRequirements_( item );
    } );

    toolbar_.setRibbonMenu( this );
}

void RibbonMenu::shutdown()
{
    for ( auto& item : RibbonSchemaHolder::schema().items )
    {
        if ( item.second.item && item.second.item->isActive() )
            item.second.item->action();
    }
    fontManager_.initFontManagerInstance( nullptr );
    ImGuiMenu::shutdown();
    RibbonIcons::free();
}

void RibbonMenu::openToolbarCustomize()
{
    toolbar_.openCustomize();
}

// we use design preset font size
void RibbonMenu::load_font( int )
{
    ImVector<ImWchar> ranges;
    ImFontGlyphRangesBuilder builder;
    addMenuFontRanges_( builder );
    builder.BuildRanges( &ranges );
    fontManager_.loadAllFonts( ranges.Data, menu_scaling() );
}

std::filesystem::path RibbonMenu::getMenuFontPath() const
{
    return fontManager_.getMenuFontPath();
}

void RibbonMenu::pinTopPanel( bool on )
{
    collapseState_ = on ? CollapseState::Pinned : CollapseState::Opened;
    fixViewportsSize_( getViewerInstance().framebufferSize.x, getViewerInstance().framebufferSize.y );
}

bool RibbonMenu::isTopPannelPinned() const
{
    return collapseState_ == CollapseState::Pinned;
}

void RibbonMenu::readQuickAccessList( const Json::Value& root )
{
    toolbar_.readItemsList( root );
}

void RibbonMenu::resetQuickAccessList()
{
    toolbar_.resetItemsList();
}

void RibbonMenu::setSceneSize( const Vector2i& size )
{
    sceneSize_ = ImVec2( float( size.x ), float( size.y ) );
    auto& viewerRef = getViewerInstance();
    fixViewportsSize_( viewerRef.framebufferSize.x, viewerRef.framebufferSize.y );
}

void RibbonMenu::updateItemStatus( const std::string& itemName )
{
    auto itemIt = RibbonSchemaHolder::schema().items.find( itemName );
    if ( itemIt == RibbonSchemaHolder::schema().items.end() )
        return;

    auto& item = itemIt->second.item;
    assert( item );
    if ( item->isActive() )
    {
        if ( item->blocking() )
        {
            // disable old blocking first
            if ( activeBlockingItem_.item && activeBlockingItem_.item != item )
                itemPressed_( activeBlockingItem_.item, true );
            activeBlockingItem_ = { item,false };
        }
        else
        {
            auto nonBlockingIt =
                std::find_if( activeNonBlockingItems_.begin(), activeNonBlockingItems_.end(), [&] ( const auto& it )
            {
                return it.item == item;
            } );
            // add if it is not already in the list
            if ( nonBlockingIt == activeNonBlockingItems_.end() )
                activeNonBlockingItems_.push_back( { item,false } );
        }
    }
    else
    {
        if ( item->blocking() )
        {
            if ( activeBlockingItem_.item && activeBlockingItem_.item == item )
                activeBlockingItem_ = {};
        }
        else
        {
            activeNonBlockingItems_.erase(
                std::remove_if( activeNonBlockingItems_.begin(), activeNonBlockingItems_.end(), [&] ( const auto& it )
            {
                return it.item == item;
            } ),
                activeNonBlockingItems_.end()
                );
        }
    }
}

void RibbonMenu::drawActiveBlockingDialog_()
{
    drawItemDialog_( activeBlockingItem_ );
    highlightBlocking_();
}

void RibbonMenu::drawActiveNonBlockingDialogs_()
{
    for ( auto& item : activeNonBlockingItems_ )
        drawItemDialog_( item );

    activeNonBlockingItems_.erase(
        std::remove_if( activeNonBlockingItems_.begin(), activeNonBlockingItems_.end(), [] ( const auto& it )
    {
        return !bool( it.item );
    } ),
        activeNonBlockingItems_.end()
        );
}

void RibbonMenu::drawSearchButton_()
{
    const auto scaling = menu_scaling();
    auto font = fontManager_.getFontByType( RibbonFontManager::FontType::Icons );
    font->Scale = 0.7f;

    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    ImGui::PushStyleColor( ImGuiCol_Button, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrab ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );

    auto absMinPos = ImGui::GetCurrentContext()->CurrentWindow->DC.CursorPos;

    float btnSize = scaling * cTopPanelAditionalButtonSize;
    ImGui::PushFont( font );
    auto pressed = ImGui::Button( "\xef\x80\x82", ImVec2( btnSize, btnSize ) );
    ImGui::PopFont();

    font->Scale = 1.0f;

    ImGui::PopStyleColor( 3 );
    ImGui::PopStyleVar( 2 );

    auto nameWindow = "##RibbonGlobalSearchPopup";
    bool popupOpened = ImGui::IsPopupOpen( nameWindow );

    // manage search popup
    if ( pressed && !popupOpened )
        ImGui::OpenPopup( nameWindow );

    if ( !popupOpened )
        return;

    if ( ImGuiWindow* menuWindow = ImGui::FindWindowByName( nameWindow ) )
        if ( menuWindow->WasActive )
        {
            ImRect frame;
            frame.Min = absMinPos;
            frame.Max = ImVec2( frame.Min.x + ImGui::GetFrameHeight(), frame.Min.y + ImGui::GetFrameHeight() );
            ImVec2 expectedSize = ImGui::CalcWindowNextAutoFitSize( menuWindow );
            menuWindow->AutoPosLastDirection = ImGuiDir_Down;
            ImRect rectOuter = ImGui::GetPopupAllowedExtentRect( menuWindow );
            ImVec2 pos = ImGui::FindBestWindowPosForPopupEx( frame.GetBL(), expectedSize, &menuWindow->AutoPosLastDirection, rectOuter, frame, ImGuiPopupPositionPolicy_ComboBox );
            ImGui::SetNextWindowPos( pos );
        }

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_Popup | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove;
    ImGui::Begin( nameWindow, NULL, window_flags );
    if ( popupOpened )
    {
        // Search line
        if ( ImGui::IsWindowAppearing() )
        {
            searchLine_.clear();
            searchResult_.clear();
            ImGui::SetKeyboardFocusHere();
        }
        float minSearchSize = 300.0f * scaling;
        ImGui::SetNextItemWidth( minSearchSize );
        if ( ImGui::InputText( "##SearchLine", searchLine_ ) )
            searchResult_ = RibbonSchemaHolder::search( searchLine_ );

        ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::Small ) );
        auto ySize = ( cSmallIconSize + 2 * cRibbonButtonWindowPaddingY ) * scaling;
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, 
                               ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive,
                               ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActive ).getUInt32() );
        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
        int uniqueBtnCounter = 0;
        for ( const auto& foundItem : searchResult_ )
        {
            if ( !foundItem.item )
                continue;
            auto pos = ImGui::GetCursorPos();
            if ( foundItem.tabIndex != -1 )
            {
                const auto& tabName = RibbonSchemaHolder::schema().tabsOrder[foundItem.tabIndex].name;
                auto label = "##SearchTabBtn" + tabName + std::to_string( ++uniqueBtnCounter );
                auto labelSize = ImGui::CalcTextSize( tabName.c_str() );
                if ( ImGui::Button( label.c_str(), ImVec2( labelSize.x + 2 * cRibbonButtonWindowPaddingX * scaling, ySize ) ) )
                {
                    changeTab_( foundItem.tabIndex );
                    ImGui::CloseCurrentPopup();
                }
                ImVec2 textPos = pos;
                textPos.x += cRibbonButtonWindowPaddingX * scaling;
                textPos.y += ( ySize - labelSize.y ) * 0.5f;
                ImGui::SetCursorPos( textPos );
                ImGui::Text( "%s", tabName.c_str() );
                ImGui::SameLine( 0.0f, cRibbonButtonWindowPaddingX * scaling + ImGui::GetStyle().ItemSpacing.x );
                ImGui::SetCursorPosX( minSearchSize * 0.25f );
                ImGui::Text( ">" );
                ImGui::SameLine( 0.0f, cRibbonButtonWindowPaddingX * scaling + ImGui::GetStyle().ItemSpacing.x );
            }
            auto width = buttonDrawer_.calcItemWidth( *foundItem.item, DrawButtonParams::SizeType::SmallText );
            DrawButtonParams params;
            params.sizeType = DrawButtonParams::SizeType::SmallText;
            params.iconSize = cSmallIconSize;
            params.itemSize.y = ySize;
            params.itemSize.x = width.baseWidth + width.additionalWidth + 2.0f * cRibbonButtonWindowPaddingX * scaling;
            ImGui::SetCursorPosY( pos.y );
            buttonDrawer_.drawButtonItem( *foundItem.item, params );
        }
        ImGui::PopStyleVar( 1 );
        ImGui::PopStyleColor( 3 );
        ImGui::PopFont();
        ImGui::EndPopup();
    }
}

void RibbonMenu::drawCollapseButton_()
{
    const auto scaling = menu_scaling();
    auto font = fontManager_.getFontByType( RibbonFontManager::FontType::Icons );
    font->Scale = 0.7f;

    float btnSize = scaling * cTopPanelAditionalButtonSize;

    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    ImGui::PushStyleColor( ImGuiCol_Button, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrab ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );

    if ( collapseState_ == CollapseState::Pinned )
    {
        ImGui::PushFont( font );
        if ( ImGui::Button( "\xef\x81\x93", ImVec2( btnSize, btnSize ) ) )
        {
            collapseState_ = CollapseState::Opened;
            fixViewportsSize_( getViewerInstance().framebufferSize.x, getViewerInstance().framebufferSize.y );
            openedTimer_ = openedMaxSecs_;
#ifndef __EMSCRIPTEN__
            asyncRequest_.reset();
#endif
        }
        ImGui::PopFont();
        if ( ImGui::IsItemHovered() )
        {
            ImGui::BeginTooltip();
            ImGui::Text( "%s", "Unpin" );
            ImGui::EndTooltip();
        }
    }
    else
    {
        ImGui::PushFont( font );
        if ( ImGui::Button( "\xef\x81\xb7", ImVec2( btnSize, btnSize ) ) )
        {
            collapseState_ = CollapseState::Pinned;
            fixViewportsSize_( getViewerInstance().framebufferSize.x, getViewerInstance().framebufferSize.y );
        }
        ImGui::PopFont();
        if ( ImGui::IsItemHovered() )
        {
            ImGui::BeginTooltip();
            ImGui::Text( "%s", "Pin" );
            ImGui::EndTooltip();
        }
    }
    font->Scale = 1.0f;

    ImGui::PopStyleColor( 3 );
    ImGui::PopStyleVar( 2 );

    if ( collapseState_ == CollapseState::Opened )
    {
        bool hovered = ImGui::IsWindowHovered(
            ImGuiHoveredFlags_ChildWindows |
            ImGuiHoveredFlags_AllowWhenBlockedByActiveItem );
        if ( hovered && openedTimer_ <= openedMaxSecs_ )
        {
#ifndef __EMSCRIPTEN__
            asyncRequest_.reset();
#endif
            openedTimer_ = openedMaxSecs_;
            collapseState_ = CollapseState::Opened;
        }
        else
        {
            openedTimer_ -= ImGui::GetIO().DeltaTime;
#ifdef __EMSCRIPTEN__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
            EM_ASM( postEmptyEvent( $0, 2 ), int( openedTimer_ * 1000 ) );
#pragma clang diagnostic pop
#else
            asyncRequest_.requestIfNotSet(
                std::chrono::system_clock::now() + std::chrono::milliseconds( std::llround( openedTimer_ * 1000 ) ),
                [] ()
            {
                CommandLoop::appendCommand( [] ()
                {
                    getViewerInstance().incrementForceRedrawFrames();
                } );
            } );
#endif
            if ( openedTimer_ <= 0.0f )
                collapseState_ = CollapseState::Closed;
        }
    }
}

void RibbonMenu::sortObjectsRecursive_( std::shared_ptr<Object> object )
{
    auto& children = object->children();
    for ( const auto& child : children )
        sortObjectsRecursive_( child );

    AppendHistory( std::make_shared<ChangeSceneObjectsOrder>( "Sort object children", object ) );
    object->sortChildren();
}

void RibbonMenu::drawHeaderQuickAccess_()
{
    const float menuScaling = menu_scaling();

    auto itemSpacing = ImVec2( cHeaderQuickAccessXSpacing * menuScaling, ( cTabHeight + cTabYOffset - cHeaderQuickAccessFrameSize ) * menuScaling * 0.5f );
    auto iconSize = cHeaderQuickAccessIconSize;
    auto itemSize = cHeaderQuickAccessFrameSize * menuScaling;

    int dropCount = 0;
    for ( const auto& item : RibbonSchemaHolder::schema().headerQuickAccessList )
    {
        auto it = RibbonSchemaHolder::schema().items.find( item );
        if ( it == RibbonSchemaHolder::schema().items.end() )
            continue;
        if ( it->second.item && it->second.item->type() == RibbonItemType::ButtonWithDrop )
            dropCount++;
    }

    const auto width = RibbonSchemaHolder::schema().headerQuickAccessList.size() * ( itemSpacing.x + itemSize ) +
        dropCount * cSmallItemDropSizeModifier * itemSize;
    const auto availableWidth = getViewerInstance().framebufferSize.x;
    if ( width * 2 > availableWidth )
        return; // dont show header quick panel if window is too small

    ImGui::SetCursorPos( itemSpacing );

    DrawButtonParams params{ DrawButtonParams::SizeType::Small, ImVec2( itemSize,itemSize ), iconSize,DrawButtonParams::RootType::Header };

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding * menuScaling );
    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::Small ) );
    for ( const auto& item : RibbonSchemaHolder::schema().headerQuickAccessList )
    {
        auto it = RibbonSchemaHolder::schema().items.find( item );
        if ( it == RibbonSchemaHolder::schema().items.end() )
        {
#ifndef __EMSCRIPTEN__
            spdlog::warn( "Plugin \"{}\" not found!", item );
#endif
            continue;
        }

        buttonDrawer_.drawButtonItem( it->second, params );
        ImGui::SameLine();
    }
    ImGui::PopFont();
    ImGui::PopStyleVar( 2 );

    ImGui::SetCursorPosX( ImGui::GetCursorPosX() - itemSpacing.x );
    ImGui::SetCursorPosY( 0.0f );
}

void RibbonMenu::drawHeaderPannel_()
{
    const float menuScaling = menu_scaling();
    ImGui::PushStyleVar( ImGuiStyleVar_TabRounding, cTabFrameRounding * menuScaling );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 0, 0 ) );

    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddRectFilled(
        ImVec2( 0, 0 ),
        ImVec2( float( getViewerInstance().framebufferSize.x ), ( cTabHeight + cTabYOffset ) * menuScaling ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::HeaderBackground ).getUInt32() );

    drawHeaderQuickAccess_();

    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::SemiBold ) );
    // TODO_store: this needs recalc only on scaling change, no need to calc each frame
    std::vector<float> textSizes( RibbonSchemaHolder::schema().tabsOrder.size() );// TODO_store: add to some store at the beginning not to calc each time  
    std::vector<float> tabSizes( RibbonSchemaHolder::schema().tabsOrder.size() );// TODO_store: add to some store at the beginning not to calc each time  
    auto summaryTabPannelSize = 2 * 12.0f * menuScaling - cTabsInterval * menuScaling; // init shift (by eye, not defined in current design maket)
    for ( int i = 0; i < tabSizes.size(); ++i )
    {
        const auto& tabStr = RibbonSchemaHolder::schema().tabsOrder[i].name;
        textSizes[i] = ImGui::CalcTextSize( tabStr.c_str() ).x;
        tabSizes[i] = std::max( textSizes[i] + cTabLabelMinPadding * 2 * menuScaling, cTabMinimumWidth * menuScaling );
        summaryTabPannelSize += ( tabSizes[i] + cTabsInterval * menuScaling );
    }
    // prepare active button
    bool needActive = hasAnyActiveItem() && toolbar_.getCurrentToolbarWidth() == 0.0f;
    float activeBtnSize = cTabHeight * menuScaling - 4 * menuScaling; // small offset from border

    // 40 - search button size (by eye)
    // 40 - collapse button size (by eye)
    // 40 - active button size (by eye)
    auto availWidth = ImGui::GetContentRegionAvail().x - ( needActive ? 3 : 2 ) * 40.0f * menuScaling;

    float scrollMax = summaryTabPannelSize - availWidth;
    bool needScroll = scrollMax > 0.0f;
    ImGui::BeginChild( "##TabsScrollHeaderWindow", ImVec2( availWidth, ( cTabYOffset + cTabHeight ) * menuScaling ) );

    auto tabsWindowWidth = availWidth;
    auto tabsWindowPosX = 0.0f;
    bool needBackBtn = needScroll && tabPanelScroll_ != 0;
    if ( needBackBtn )
        scrollMax += ( cTopPanelScrollBtnSize + 2 * cTabsInterval ) * menuScaling;// size of back btn
    if ( tabPanelScroll_ > scrollMax )
        tabPanelScroll_ = scrollMax;
    bool needFwdBtn = needScroll && tabPanelScroll_ != scrollMax;
    if ( !needScroll )
        tabPanelScroll_ = 0.0f;
    if ( needBackBtn )
    {
        tabsWindowWidth -= ( cTopPanelScrollBtnSize + 2 * cTabsInterval ) * menuScaling;// back scroll btn size
        tabsWindowPosX = ( cTopPanelScrollBtnSize + 2 * cTabsInterval ) * menuScaling;
    }
    if ( needFwdBtn )
    {
        tabsWindowWidth -= ( cTopPanelScrollBtnSize + 2 * cTabsInterval ) * menuScaling;// forward scroll btn size
    }
    if ( tabsWindowWidth <= 0.0f )// <=0.0 - special values that should be ignored
        tabsWindowWidth = 1.0f;
    if ( needBackBtn )
    {
        ImGui::SetCursorPosX( cTabsInterval * menuScaling );
        const float btnSize = 0.5f * fontManager_.getFontSizeByType( RibbonFontManager::FontType::Icons );
        if ( buttonDrawer_.drawTabArrawButton( "\xef\x81\x88", ImVec2( cTopPanelScrollBtnSize * menuScaling, ( cTabYOffset + cTabHeight ) * menuScaling ), btnSize ) )
        {
            tabPanelScroll_ -= cTopPanelScrollStep * menuScaling;
            if ( tabPanelScroll_ < 0.0f )
                tabPanelScroll_ = 0.0f;
        }
        ImGui::SameLine();
    }
    ImGui::SetCursorPosX( tabsWindowPosX );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0, 0 ) );
    ImGui::BeginChild( "##TabsHeaderWindow", ImVec2( tabsWindowWidth, ( cTabYOffset + cTabHeight ) * menuScaling ) );
    ImGui::PopStyleVar();
    auto window = ImGui::GetCurrentContext()->CurrentWindow;

    auto basePos = window->Pos;
    if ( needScroll )
    {
        basePos.x -= tabPanelScroll_;
    }
    basePos.x += 12.0f * menuScaling;// temp hardcoded offset
    basePos.y = cTabYOffset * menuScaling - 1;// -1 due to ImGui::TabItemBackground internal offset
    for ( int i = 0; i < RibbonSchemaHolder::schema().tabsOrder.size(); ++i )
    {
        const auto& tabStr = RibbonSchemaHolder::schema().tabsOrder[i].name;
        const auto& tabWidth = tabSizes[i];
        ImVec2 tabBbMaxPoint( basePos.x + tabWidth, basePos.y + cTabHeight * menuScaling + 2 ); // +2 due to TabItemBackground internal offset
        ImRect tabRect( basePos, tabBbMaxPoint );
        std::string strId = "##" + tabStr + "TabId"; // TODO_store: add to some store at the beginning not to calc each time
        auto tabId = window->GetID( strId.c_str() );
        ImGui::ItemAdd( tabRect, tabId );
        bool hovered, held;
        bool pressed = ImGui::ButtonBehavior( tabRect, tabId, &hovered, &held );
        if ( pressed )
            changeTab_( i );

        if ( activeTabIndex_ == i || hovered || pressed )
        {
            Color tabRectColor; 
            if ( activeTabIndex_ == i )
            {
                if ( pressed )
                    tabRectColor = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActiveClicked );
                else if ( hovered )
                    tabRectColor = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActiveHovered );
                else
                    tabRectColor = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActive );
            }
            else
            {
                if ( pressed )
                    tabRectColor = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabClicked );
                else
                    tabRectColor = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered );
            }
            ImGui::TabItemBackground( window->DrawList, tabRect, 0, tabRectColor.getUInt32() );
        }
        ImGui::SetCursorPosX( basePos.x + ( tabWidth - textSizes[i] ) * 0.5f );
        // "4.0f * scaling" eliminates shift of the font
        ImGui::SetCursorPosY( 2 * cTabYOffset * menuScaling + 4.0f * menuScaling );

        if ( activeTabIndex_ == i )
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActiveText ).getUInt32() );
        else
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText ).getUInt32() );
        ImGui::RenderText( ImGui::GetCursorPos(), tabStr.c_str(), tabStr.c_str() + tabStr.size(), false );
        ImGui::PopStyleColor();

        basePos.x += ( tabWidth + cTabsInterval * menuScaling );
    }
    ImGui::Dummy( ImVec2( 0, 0 ) );
    ImGui::EndChild();
    if ( needFwdBtn )
    {
        ImGui::SameLine();
        ImGui::SetCursorPosX( ImGui::GetCursorPosX() + cTabsInterval * menuScaling );
        const float btnSize = 0.5f * fontManager_.getFontSizeByType( RibbonFontManager::FontType::Icons );
        if ( buttonDrawer_.drawTabArrawButton( "\xef\x81\x91", ImVec2( cTopPanelScrollBtnSize * menuScaling, ( cTabYOffset + cTabHeight ) * menuScaling ), btnSize ) )
        {
            if ( !needFwdBtn )
                tabPanelScroll_ += tabsWindowPosX * menuScaling;//size of back btn
            tabPanelScroll_ += cTopPanelScrollStep * menuScaling;
            if ( tabPanelScroll_ > scrollMax )
                tabPanelScroll_ = scrollMax;
        }
    }
    ImGui::EndChild();
    ImGui::PopFont();

    ImGui::PopStyleVar( 2 );
    const float separateLinePos = ( cTabYOffset + cTabHeight ) * menuScaling;
    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddLine( ImVec2( 0, separateLinePos ), ImVec2( float( getViewerInstance().framebufferSize.x ), separateLinePos ),
                                                                  ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::HeaderSeparator ).getUInt32() );

    if ( needActive )
    {
        ImGui::SetCursorPos( ImVec2( float( getViewerInstance().framebufferSize.x ) - 110.0f * menuScaling, cTabYOffset * menuScaling ) );
        drawActiveListButton_( activeBtnSize );
    }

    ImGui::SetCursorPos( ImVec2( float( getViewerInstance().framebufferSize.x ) - 70.0f * menuScaling, cTabYOffset* menuScaling ) );
    drawSearchButton_();

    ImGui::SetCursorPos( ImVec2( float( getViewerInstance().framebufferSize.x ) - 30.0f * menuScaling, cTabYOffset * menuScaling ) );
    drawCollapseButton_();
}

void RibbonMenu::drawActiveListButton_( float btnSize )
{
    auto activeListIt = RibbonSchemaHolder::schema().items.find( "Active Plugins List" );
    if ( activeListIt != RibbonSchemaHolder::schema().items.end() )
    {
        setActiveListPos( ImGui::GetCursorScreenPos() );
        CustomButtonParameters cParams;
        cParams.iconType = RibbonIcons::IconType::RibbonItemIcon;
        cParams.pushColorsCb = [] ( bool enabled, bool )->int
        {
            if ( !enabled )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TextDisabled ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_Button, Color( 0, 0, 0, 0 ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarHovered ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarClicked ).getUInt32() );
            }
            else
            {
                ImGui::PushStyleColor( ImGuiCol_Text, Color::white().getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_Button, Color( 60, 169, 20, 255 ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_ButtonHovered, Color( 60, 169, 20, 200 ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_ButtonActive, Color( 60, 169, 20, 255 ).getUInt32() );
            }
            return 4;
        };
        const ImVec2 itemSize = { btnSize, btnSize };
        DrawButtonParams params{ DrawButtonParams::SizeType::Small, itemSize, cMiddleIconSize,DrawButtonParams::RootType::Toolbar };
        buttonDrawer_.drawCustomButtonItem( activeListIt->second, cParams, params );
    }
}


void RibbonMenu::drawActiveList_()
{
    auto pressed = activeListPressed_;
    activeListPressed_ = false;

    auto nameWindow = "##ActiveList";
    bool popupOpened = ImGui::IsPopupOpen( nameWindow );

    // manage search popup
    if ( pressed && !popupOpened )
        ImGui::OpenPopup( nameWindow );

    if ( !popupOpened )
        return;
    auto scaling = menu_scaling();
    if ( ImGuiWindow* menuWindow = ImGui::FindWindowByName( nameWindow ) )
        if ( menuWindow->WasActive )
        {
            ImRect frame;
            frame.Min = activeListPos_;
            frame.Min.x -= 6 * scaling;
            frame.Min.y += 10 * scaling;
            frame.Max = ImVec2( frame.Min.x + ImGui::GetFrameHeight(), frame.Min.y + ImGui::GetFrameHeight() );
            ImVec2 expectedSize = ImGui::CalcWindowNextAutoFitSize( menuWindow );
            menuWindow->AutoPosLastDirection = ImGuiDir_Down;
            ImRect rectOuter = ImGui::GetPopupAllowedExtentRect( menuWindow );
            ImVec2 pos = ImGui::FindBestWindowPosForPopupEx( frame.GetBL(), expectedSize, &menuWindow->AutoPosLastDirection, rectOuter, frame, ImGuiPopupPositionPolicy_ComboBox );
            ImGui::SetNextWindowPos( pos );
        }

    ImGuiWindowFlags window_flags = 
        ImGuiWindowFlags_AlwaysAutoResize | 
        ImGuiWindowFlags_Popup | 
        ImGuiWindowFlags_NoTitleBar | 
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoSavedSettings | 
        ImGuiWindowFlags_NoMove;
    ImGui::PushStyleVar( ImGuiStyleVar_PopupBorderSize, 0.0f );
    ImGui::PushStyleColor( ImGuiCol_PopupBg, ImVec4( 0, 0, 0, 0 ) );
    ImGui::Begin( nameWindow, NULL, window_flags );
    if ( popupOpened )
    {
        bool closeBlocking = false;
        std::vector<bool> closeNonBlocking( activeNonBlockingItems_.size(), false );

        auto winPadding = ImVec2( 6 * scaling, 4 * scaling );
        auto itemSpacing = ImVec2( 10 * scaling, 4 * scaling );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, winPadding );
        ImGui::PushStyleVar( ImGuiStyleVar_ChildRounding, 4 * scaling );
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );

        ImVec2 btnSize = ImVec2( 56.0f * scaling, 24.0f * scaling );
        float maxSize = 0.0f;
        auto getItemCaption = [] ( const std::string& name )->const std::string&
        {
            auto it = RibbonSchemaHolder::schema().items.find( name );
            if ( it == RibbonSchemaHolder::schema().items.end() )
                return name;
            return  it->second.caption.empty() ? name : it->second.caption;
        };

        auto sbFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::SemiBold );
        if ( sbFont )
            ImGui::PushFont( sbFont );
        if ( activeBlockingItem_.item )
            maxSize = ImGui::CalcTextSize( getItemCaption( activeBlockingItem_.item->name() ).c_str() ).x;
        for ( const auto& nonBlockItem : activeNonBlockingItems_ )
        {
            auto size = ImGui::CalcTextSize( getItemCaption( nonBlockItem.item->name() ).c_str() ).x;
            if ( size > maxSize )
                maxSize = size;
        }
        if ( sbFont )
            ImGui::PopFont();

        auto blockSize = ImVec2( 2 * winPadding.x + maxSize + 2 * ImGui::GetStyle().ItemSpacing.x + btnSize.x,
            btnSize.y + winPadding.y * 2 );
        auto dotShift = ( blockSize.y - 2 * scaling ) * 0.5f;
        blockSize.x = blockSize.x - winPadding.x + dotShift;

        auto drawItem = [&] ( const std::shared_ptr<RibbonMenuItem>& item, bool& close )
        {
            if ( !item )
                return;
            const auto& name = getItemCaption( item->name() );
            auto childName = "##CloseItemBlock" + item->name();

            ImGui::PushStyleColor( ImGuiCol_ChildBg, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Background ).getUInt32() );
            ImGui::BeginChild( childName.c_str(), blockSize, true,
                ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse );
            ImGui::PopStyleColor();
            if ( sbFont )
                ImGui::PushFont( sbFont );
            auto center = ImGui::GetCursorScreenPos();
            center.x += dotShift - winPadding.x;
            center.y += dotShift - winPadding.y;
            ImGui::GetWindowDrawList()->AddCircleFilled( center, 2 * scaling, Color( 60, 169, 20, 255 ).getUInt32() );
            
            ImGui::SetCursorPosX( dotShift + 2 * scaling + itemSpacing.x );
            auto savedPos = ImGui::GetCursorPosY();
            ImGui::SetCursorPosY( 0.5f * ( blockSize.y - ImGui::GetFontSize() ) );
            ImGui::Text( "%s", name.c_str() );
            if ( sbFont )
                ImGui::PopFont();
            ImGui::SameLine( blockSize.x - btnSize.x - winPadding.x );
            ImGui::SetCursorPosY( savedPos );
            if ( UI::button( "Close", btnSize ) )
                close = true;
            ImGui::EndChild();
        };

        drawItem( activeBlockingItem_.item, closeBlocking );
        for ( int i = 0; i < activeNonBlockingItems_.size(); ++i )
        {
            bool close{ false };
            drawItem( activeNonBlockingItems_[i].item, close );
            closeNonBlocking[i] = close;
        }

        if ( !activeBlockingItem_.item && activeNonBlockingItems_.empty() )
            ImGui::CloseCurrentPopup();

        ImGui::PopStyleVar( 3 );
        ImGui::EndPopup();

        if ( closeBlocking )
            itemPressed_( activeBlockingItem_.item, true );
        for ( int i = 0; i < activeNonBlockingItems_.size(); ++i )
            if ( closeNonBlocking[i] )
                itemPressed_( activeNonBlockingItems_[i].item, true );
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

bool RibbonMenu::drawGroupUngroupButton_( const std::vector<std::shared_ptr<Object>>& selected )
{
    bool someChanges = false;
    if ( selected.empty() )
        return someChanges;

    Object* parentObj = selected[0]->parent();
    bool canGroup = parentObj != nullptr && selected.size() >= 2;
    for ( int i = 1; canGroup && i < selected.size(); ++i )
    {
        if ( selected[i]->parent() != parentObj )
            canGroup = false;
    }

    if ( canGroup && UI::button( "Group", Vector2f( -1, 0 ) ) )
    {
        someChanges |= true;
        std::shared_ptr<Object> group = std::make_shared<Object>();
        group->setAncillary( false );
        group->setName( "Group" );

        SCOPED_HISTORY( "Group objects" );
        AppendHistory<ChangeSceneAction>( "Add object", group, ChangeSceneAction::Type::AddObject );
        parentObj->addChild( group );
        group->select( true );
        for ( int i = 0; i < selected.size(); ++i )
        {
            // for now do it by one object
            AppendHistory<ChangeSceneAction>( "Remove object", selected[i], ChangeSceneAction::Type::RemoveObject );
            selected[i]->detachFromParent();
            AppendHistory<ChangeSceneAction>( "Remove object", selected[i], ChangeSceneAction::Type::AddObject );
            group->addChild( selected[i] );
            selected[i]->select( false );
        }
    }

    bool canUngroup = selected.size() == 1;
    if ( canUngroup )
    {
        canUngroup = false;
        for ( const auto& child : selected[0]->children() )
        {
            if ( !child->isAncillary() )
            {
                canUngroup = true;
                break;
            }
        }
    }
        canUngroup = !selected[0]->children().empty();
    if ( canUngroup && UI::button( "Ungroup", Vector2f( -1, 0 ) ) )
    {
        someChanges |= true;
        auto children = selected[0]->children();
        SCOPED_HISTORY( "Ungroup objects" );
        selected[0]->select( false );
        for ( int i = 0; i < children.size(); ++i )
        {
            if ( children[i]->isAncillary() )
                continue;
            // for now do it by one object
            AppendHistory<ChangeSceneAction>( "Remove object", children[i], ChangeSceneAction::Type::RemoveObject );
            children[i]->detachFromParent();
            AppendHistory<ChangeSceneAction>( "Add object", children[i], ChangeSceneAction::Type::AddObject );
            parentObj->addChild( children[i] );
            children[i]->select( true );
        }
        auto ptr = std::dynamic_pointer_cast< VisualObject >( selected[0] );
        if ( !ptr && selected[0]->children().empty() )
        {
            AppendHistory<ChangeSceneAction>( "Remove object", selected[0], ChangeSceneAction::Type::RemoveObject );
            selected[0]->detachFromParent();
        }
    }

    return someChanges;
}

void RibbonMenu::cloneTree( const std::vector<std::shared_ptr<Object>>& selectedObjects )
{
    const std::regex pattern( R"(.* Clone(?:| \([0-9]+\))$)" );
    SCOPED_HISTORY( "Clone objects" );
    for ( const auto& obj : selectedObjects )
    {
        if ( !obj )
            continue;
        auto cloneObj = obj->cloneTree();
        AppendHistory<ChangeObjectSelectedAction>( "unselect base obj", obj );
        obj->select( false );
        AppendHistory<ChangeObjectVisibilityAction>( "make base obj invisible", obj );
        obj->setVisible( false );
        auto name = obj->name();
        if ( std::regex_match( name, pattern ) )
        {
            auto endBracPos = name.rfind( ')' );
            if ( endBracPos != int( name.length() ) - 1 )
            {
                name += " (2)";
            }
            else
            {
                auto startNumPos = name.rfind( '(' ) + 1;
                auto numStr = name.substr( startNumPos, endBracPos - startNumPos );
                int num = std::atoi( numStr.c_str() );
                name = name.substr( 0, startNumPos - 1 ) + "(" + std::to_string( num + 1 ) + ")";
            }
        }
        else
        {
            name += " Clone";
        }
        cloneObj->setName( name );
        AppendHistory<ChangeSceneAction>( "Add cloned obj", cloneObj, ChangeSceneAction::Type::AddObject );
        obj->parent()->addChild( cloneObj );
    }
}

void RibbonMenu::cloneSelectedPart( const std::shared_ptr<Object>& object )
{
    std::shared_ptr<VisualObject> newObj;
    std::string name;
    if ( auto selectedMesh = std::dynamic_pointer_cast< ObjectMesh >( object ) )
    {
        if ( !selectedMesh->mesh() )
            return;
        const auto& curMesh = *selectedMesh->mesh();
        std::shared_ptr<ObjectMesh> objMesh = std::make_shared<ObjectMesh>();
        objMesh->setMesh( std::make_shared<Mesh>( curMesh.cloneRegion( selectedMesh->getSelectedFaces() ) ) );
        newObj = objMesh;
        name = "ObjectMesh";
    }
    else if ( auto selectedPoints = std::dynamic_pointer_cast< ObjectPoints >( object ) )
    {
        if ( !selectedPoints->pointCloud() )
            return;
        PointCloud newPointCloud;
        const auto& curPointCloud = *selectedPoints->pointCloud();
        newPointCloud.addPartByMask( curPointCloud, selectedPoints->getSelectedPoints() );
        std::shared_ptr<ObjectPoints> objPoints = std::make_shared<ObjectPoints>();
        objPoints->setPointCloud( std::make_shared<PointCloud>( std::move( newPointCloud ) ) );
        newObj = objPoints;
        name = "ObjectPoints";
    }

    newObj->setName( object->name() + " Partial" );
    newObj->setXf( object->xf() );
    AppendHistory<ChangeSceneAction>( "Selection to New object: add " + name, newObj, ChangeSceneAction::Type::AddObject );
    object->parent()->addChild( newObj );
}

bool RibbonMenu::drawCloneButton_( const std::vector<std::shared_ptr<Object>>& selected )
{
    bool someChanges = false;
    if ( selected.empty() )
        return someChanges;

    if ( UI::button( "Clone", Vector2f( -1, 0 ) ) )
    {
        cloneTree( selected );
        someChanges = true;
    }

    return someChanges;
}

bool RibbonMenu::drawCloneSelectionButton_( const std::vector<std::shared_ptr<Object>>& selected )
{
    bool someChanges = false;
    if ( selected.size() != 1 )
        return someChanges;
    auto objMesh = selected[0]->asType<ObjectMesh>();
    auto objPoints = selected[0]->asType<ObjectPoints>();
    if ( ( objMesh && objMesh->getSelectedFaces().any() ) ||
         ( objPoints && objPoints->getSelectedPoints().any() ) )
    {
        if ( UI::button( "Clone Selection", Vector2f( -1, 0 ) ) )
        {
            cloneSelectedPart( selected[0] );
            someChanges = true;
        }
    }

    return someChanges;
}

void RibbonMenu::drawBigButtonItem_( const MenuItemInfo& item )
{
    auto width = buttonDrawer_.calcItemWidth( item, DrawButtonParams::SizeType::Big );

    auto availReg = ImGui::GetContentRegionAvail();

    const auto& style = ImGui::GetStyle();
    ImVec2 itemSize = ImVec2( width.baseWidth, availReg.y - 2 * style.WindowPadding.y );

    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + availReg.y * 0.5f - itemSize.y * 0.5f - ImGui::GetStyle().CellPadding.y * 0.5f );

    buttonDrawer_.drawButtonItem( item, { DrawButtonParams::SizeType::Big,itemSize,cBigIconSize } );
}

void RibbonMenu::drawSmallButtonsSet_( const std::vector<std::string>& group, int setFrontIndex, int setLength, bool withText )
{
    assert( setFrontIndex >= 0 );
    assert( setLength <= 3 );
    assert( setLength > 0 );
    assert( setFrontIndex + setLength <= group.size() );

    const auto& style = ImGui::GetStyle();

    const float cIconSize = cSmallIconSize * menu_scaling();

    float maxSetWidth = 0.0f;
    std::array<RibbonButtonDrawer::ButtonItemWidth, 3> widths;
    std::array<const MenuItemInfo*, 3> items{ nullptr,nullptr ,nullptr };
    auto type = withText ? DrawButtonParams::SizeType::SmallText : DrawButtonParams::SizeType::Small;
    for ( int i = setFrontIndex; i < setFrontIndex + setLength; ++i )
    {
        auto it = RibbonSchemaHolder::schema().items.find( group[i] );
        if ( it == RibbonSchemaHolder::schema().items.end() )
            continue; // TODO: assert or log

        widths[i - setFrontIndex] = buttonDrawer_.calcItemWidth( it->second, type );
        auto sumWidth = widths[i - setFrontIndex].baseWidth + widths[i - setFrontIndex].additionalWidth;
        items[i - setFrontIndex] = &it->second;
        if ( sumWidth > maxSetWidth )
            maxSetWidth = sumWidth;
    }
    ImVec2 setSize;

    auto availReg = ImGui::GetContentRegionAvail();

    setSize.x = maxSetWidth;

    setSize.y = availReg.y - 2 * style.WindowPadding.y;

    ImVec2 smallItemSize;
    smallItemSize.x = setSize.x;
    smallItemSize.y = std::min( cIconSize + 2 * style.WindowPadding.y, setSize.y / 3.0f );
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + availReg.y * 0.5f - setSize.y * 0.5f - ImGui::GetStyle().CellPadding.y * 0.5f );

    float spaceY = ( setSize.y - 3.0f * smallItemSize.y ) * 0.5f;

    ImGui::BeginChild( ( "##SmallSet" + group[setFrontIndex] ).c_str(), setSize );
    auto basePosY = ImGui::GetCursorPosY();
    for ( int i = setFrontIndex; i < setFrontIndex + setLength; ++i )
    {
        smallItemSize.x = widths[i - setFrontIndex].baseWidth;
        if ( withText )
            smallItemSize.x += widths[i - setFrontIndex].additionalWidth;

        ImGui::SetCursorPosY( basePosY + float( i - setFrontIndex ) * ( spaceY + smallItemSize.y ) );
        buttonDrawer_.drawButtonItem( *items[i - setFrontIndex], { type,smallItemSize,cSmallIconSize } );
    }

    ImGui::EndChild();
}


RibbonMenu::DrawTabConfig RibbonMenu::setupItemsGroupConfig_( const std::vector<std::string>& groupsInTab,
                                                              const std::string& tabName )
{
    DrawTabConfig res( groupsInTab.size() );
    const auto& style = ImGui::GetStyle();
    const float screenWidth = float( getViewerInstance().framebufferSize.x ) - ImGui::GetCursorScreenPos().x -
        ( float( groupsInTab.size() ) + 1.0f ) * menu_scaling();
    std::vector<float> groupWidths( groupsInTab.size() );
    float sumWidth = 0.0f;

    auto calcGroupWidth = [&] ( const MenuItemsList& items, DrawGroupConfig config )->float
    {
        assert( config.numBig + config.numSmallText + config.numSmall == items.size() );
        float resWidth = 0.0f;
        for ( int i = 0; i < items.size();)
        {
            if ( config.numBig > 0 )
            {
                auto itemIt = RibbonSchemaHolder::schema().items.find( items[i] );
                ++i;
                --config.numBig;
                if ( itemIt == RibbonSchemaHolder::schema().items.end() )
                    continue; // TODO: asserts or log
                resWidth += buttonDrawer_.calcItemWidth( itemIt->second, DrawButtonParams::SizeType::Big ).baseWidth;
                resWidth += style.ItemSpacing.x;
                continue;
            }
            else
            {
                bool smallText = config.numSmallText > 0;
                assert( smallText || config.numSmall > 0 );
                auto& num = smallText ? config.numSmallText : config.numSmall;
                int n = std::min( 3, num );
                float maxWidth = 0.0f;
                for ( int j = i; j < i + n; ++j )
                {
                    auto itemIt = RibbonSchemaHolder::schema().items.find( items[j] );
                    if ( itemIt == RibbonSchemaHolder::schema().items.end() )
                        continue; // TODO: asserts or log
                    auto width = buttonDrawer_.calcItemWidth( itemIt->second,
                                                              smallText ?
                                                              DrawButtonParams::SizeType::SmallText :
                                                              DrawButtonParams::SizeType::Small );
                    maxWidth = std::max( width.baseWidth + width.additionalWidth, maxWidth );
                }
                resWidth += maxWidth;
                i += n;
                num -= n;
                resWidth += style.ItemSpacing.x;
                continue;
            }
        }
        resWidth += 2.0f * style.CellPadding.x;
        resWidth -= style.ItemSpacing.x; // remove one last spacing
        return resWidth;
    };

    for ( int i = 0; i < groupsInTab.size(); ++i )
    {
        auto groupIt = RibbonSchemaHolder::schema().groupsMap.find( tabName + groupsInTab[i] );
        if ( groupIt == RibbonSchemaHolder::schema().groupsMap.end() )
            continue; // TODO: asserts or log

        res[i].numBig = int( groupIt->second.size() );

        groupWidths[i] = calcGroupWidth( groupIt->second, res[i] );
        sumWidth += groupWidths[i];
    }
    while ( sumWidth >= screenWidth )
    {
        std::vector<int> resizableGroups;
        resizableGroups.reserve( groupWidths.size() );
        for ( int i = 0; i < res.size(); ++i )
            if ( res[i].numBig + res[i].numSmallText > 0 )
                resizableGroups.push_back( i );
        if ( resizableGroups.empty() )
            break;

        auto maxElemIt = std::max_element( resizableGroups.begin(), resizableGroups.end(),
                                           [&] ( int a, int b )
        {
            if ( res[a].numBig == res[b].numBig )
                return res[a].numSmallText < res[b].numSmallText;
            return res[a].numBig < res[b].numBig;
        } );
        int maxElemId = *maxElemIt;
        auto& config = res[maxElemId];
        if ( config.numBig > 0 )
        {
            int n = std::min( config.numBig, 3 );
            config.numBig -= n;
            config.numSmallText += n;
        }
        else if ( config.numSmallText > 0 )
        {
            int n = config.numSmallText % 3;
            if ( n == 0 )
                n = 3;
            config.numSmallText -= n;
            config.numSmall += n;
        }
        sumWidth = 0;
        for ( int i = 0; i < res.size(); ++i )
        {
            auto groupIt = RibbonSchemaHolder::schema().groupsMap.find( tabName + groupsInTab[i] );
            if ( groupIt == RibbonSchemaHolder::schema().groupsMap.end() )
                continue; // TODO: asserts or log
            groupWidths[i] = calcGroupWidth( groupIt->second, res[i] );
            sumWidth += groupWidths[i];
        }
    }
    return res;
}

void RibbonMenu::setupItemsGroup_( const std::vector<std::string>& groupsInTab, const std::string& tabName )
{
    for ( const auto& g : groupsInTab )
    {
        ImGui::TableSetupColumn( ( g + "##" + tabName ).c_str(), 0 );
    }
    ImGui::TableSetupColumn( ( "##fictiveGroup" + tabName ).c_str(), 0 );
}

void RibbonMenu::drawItemsGroup_( const std::string& tabName, const std::string& groupName,
                                  DrawGroupConfig config ) // copy here for easier usage
{
    auto itemSpacing = ImGui::GetStyle().ItemSpacing;
    itemSpacing.y = menu_scaling();
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding,
                         ImVec2( cRibbonButtonWindowPaddingX * menu_scaling(), cRibbonButtonWindowPaddingY * menu_scaling() ) );

    auto groupIt = RibbonSchemaHolder::schema().groupsMap.find( tabName + groupName );
    if ( groupIt == RibbonSchemaHolder::schema().groupsMap.end() )
        return; // TODO: asserts or log
    auto defaultYPos = ImGui::GetCursorPosY();
    auto size = groupIt->second.size();
    for ( int i = 0; i < size; )
    {
        const auto& item = groupIt->second[i];
        auto it = RibbonSchemaHolder::schema().items.find( item );
        if ( it == RibbonSchemaHolder::schema().items.end() )
        {
            ++i;
            assert( false );
            continue;
        }

        ImGui::SetCursorPosY( defaultYPos - itemSpacing.y );
        if ( config.numBig > 0 )
        {
            drawBigButtonItem_( it->second );
            config.numBig--;
            i++;
            if ( i < size )
                ImGui::SameLine();
            continue;
        }
        else
        {
            bool withText = config.numSmallText > 0;
            auto& num = withText ? config.numSmallText : config.numSmall;
            assert( withText || config.numSmall > 0 );
            int n = std::min( 3, num );
            drawSmallButtonsSet_( groupIt->second, i, n, withText );
            num -= n;
            i += n;
            if ( i < size )
                ImGui::SameLine();
            continue;
        }
    }
    ImGui::PopStyleVar( 2 );
}

void RibbonMenu::itemPressed_( const std::shared_ptr<RibbonMenuItem>& item, bool available )
{
    bool wasActive = item->isActive();
    // take name before, because item can become invalid during `action`
    auto name = item->name();
    if ( !wasActive && available && ( activeBlockingItem_.item && item->blocking() ) )
    {
        spdlog::info( "Cannot activate item: \"{}\", Active: \"{}\"", name, activeBlockingItem_.item->name() );
        blockingHighlightTimer_ = 2.0f;
        return;
    }
    if ( !wasActive && !available )
        return;
    ImGui::CloseCurrentPopup();
    bool stateChanged = item->action();
    if ( !stateChanged )
        spdlog::info( "Action item: \"{}\"", name );
    else
        spdlog::info( "{} item: \"{}\"", wasActive ? std::string( "Deactivated" ) : std::string( "Activated" ), name );
}

void RibbonMenu::changeTab_( int newTab )
{
    int oldTab = activeTabIndex_;
    if ( oldTab != newTab )
    {
        activeTabIndex_ = newTab;
        tabChangedSignal( oldTab, newTab );
    }
    if ( collapseState_ == CollapseState::Closed )
        collapseState_ = CollapseState::Opened;
}

std::string RibbonMenu::getRequirements_( const std::shared_ptr<RibbonMenuItem>& item ) const
{
    return item->isAvailable( selectedObjectsCache_ );
}

void RibbonMenu::drawSceneListButtons_()
{
    auto menuScaling = menu_scaling();
    const float size = ( cMiddleIconSize + 8.f ) * menuScaling;
    const ImVec2 smallItemSize = { size, size };

    const DrawButtonParams params{ DrawButtonParams::SizeType::Small, smallItemSize, cMiddleIconSize,DrawButtonParams::RootType::Toolbar };

    auto font = fontManager_.getFontByType( RibbonFontManager::FontType::Small );
    //font->Scale = 0.75f;
    ImGui::PushFont( font );
    for ( const auto& item : RibbonSchemaHolder::schema().sceneButtonsList )
    {
        auto it = RibbonSchemaHolder::schema().items.find( item );
        if ( it == RibbonSchemaHolder::schema().items.end() )
        {
#ifndef __EMSCRIPTEN__
            spdlog::warn( "Plugin \"{}\" not found!", item ); // TODO don't flood same message
#endif
            continue;
        }

        buttonDrawer_.drawButtonItem( it->second, params );
        ImGui::SameLine();
    }
    ImGui::NewLine();
    ImGui::PopFont();
    const float separateLinePos = ImGui::GetCursorScreenPos().y;
    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddLine( ImVec2( 0, separateLinePos ), ImVec2( float( sceneSize_.x ), separateLinePos ),
                                                                  ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Borders ).getUInt32() );
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ImGui::GetStyle().ItemSpacing.y + 1.0f );
}

void RibbonMenu::drawWelcomeWindow_()
{
    static bool drawWelcome = false; // change to true when we have real welcome screen
    if ( !drawWelcome )
        return;
    
    // change this number to force draw welcome screen once on new release
    constexpr int cShowWelcomeTS = 0; 
    const auto& settingsManager = getViewerInstance().getViewportSettingsManager();
    if ( !settingsManager )
    {
        drawWelcome = false;
        return;
    }
    static const int currentWelcomeTS = settingsManager->loadInt( "cWelcomeScreenTS", 0 );
    if ( currentWelcomeTS == cShowWelcomeTS )
    {
        drawWelcome = false;
        return;
    }
    else
    {
        settingsManager->saveInt( "cWelcomeScreenTS", cShowWelcomeTS );
    }

    auto scaling = menu_scaling();
    const ImVec2 windowSize{ MR::cModalWindowWidth * scaling, -1 };
    const float posX = 0.5f * ( getViewerInstance().framebufferSize.x - windowSize.x );
    const float posY = 0.5f * ( getViewerInstance().framebufferSize.y - windowSize.x * 0.5f );
    ImGui::SetNextWindowPos( ImVec2( posX, posY ), ImGuiCond_Appearing );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * scaling, cModalWindowPaddingY * scaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 2.0f * cDefaultItemSpacing * scaling, 3.0f * cDefaultItemSpacing * scaling } );
    if ( ImGui::Begin( "##wlcomeScreen", nullptr,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        auto headerFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Headline );
        if ( headerFont )
            ImGui::PushFont( headerFont );

        const auto headerWidth = ImGui::CalcTextSize( "Welcome" ).x;

        ImGui::SetCursorPosX( ( windowSize.x - headerWidth ) * 0.5f );
        ImGui::Text( "Welcome" );

        if ( headerFont )
            ImGui::PopFont();

        const float textWidth = ImGui::CalcTextSize( "Hello!" ).x;
        ImGui::SetCursorPosX( ( windowSize.x - textWidth ) * 0.5f );
        ImGui::Text( "Hello!" );

        const auto style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * scaling } );
        if ( UI::button( "Okay", Vector2f( -1, 0 ) ) || ImGui::IsKeyPressed( ImGuiKey_Enter ) ||
           ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) ) )
        {
            drawWelcome = false;
        }
        ImGui::PopStyleVar();
        ImGui::End();
    }
    ImGui::PopStyleVar( 2 );
}

void RibbonMenu::readMenuItemsStructure_()
{
    RibbonSchemaLoader loader;
    loader.loadSchema();
    toolbar_.resetItemsList();
}

void RibbonMenu::postResize_( int width, int height )
{
    ImGuiMenu::postResize_( width, height );
    fixViewportsSize_( width, height );
}

void RibbonMenu::postRescale_( float x, float y )
{
    ImGuiMenu::postRescale_( x, y );
    buttonDrawer_.setScaling( menu_scaling() );
    toolbar_.setScaling( menu_scaling() );
    fixViewportsSize_( Viewer::instanceRef().framebufferSize.x, Viewer::instanceRef().framebufferSize.y );

    RibbonSchemaLoader loader;
    loader.recalcItemSizes();
}

void RibbonMenu::drawItemDialog_( DialogItemPtr& itemPtr )
{
    if ( itemPtr.item )
    {
        auto statePlugin = std::dynamic_pointer_cast< StateBasePlugin >( itemPtr.item );
        if ( statePlugin && statePlugin->isEnabled() )
        {
            statePlugin->preDrawUpdate();
            statePlugin->drawDialog( menu_scaling(), ImGui::GetCurrentContext() );

            if ( !itemPtr.dialogPositionFixed )
            {
                itemPtr.dialogPositionFixed = true;
                auto* window = ImGui::FindWindowByName( itemPtr.item->name().c_str() ); // this function is hidden in imgui_internal.h
                // viewer->framebufferSize.x here because ImGui use screen space
                if ( window )
                {
                    ImVec2 pos = ImVec2( viewer->framebufferSize.x - window->Size.x, float( topPanelOpenedHeight_ - 1.0f ) * menu_scaling() );
                    ImGui::SetWindowPos( window, pos, ImGuiCond_Always );
                }
            }

            if ( !statePlugin->dialogIsOpen() )
                itemPressed_( itemPtr.item, true );
            else if ( prevFrameObjectsCache_ != selectedObjectsCache_ )
                statePlugin->updateSelection( selectedObjectsCache_ );
        }
    }
}

void RibbonMenu::drawRibbonSceneList_()
{
    const auto allObj = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selectable );
    auto selectedObjs = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );

    const auto scaling = menu_scaling();
    // Define next window position + size
    auto& viewerRef = Viewer::instanceRef();
    ImGui::SetWindowPos( "RibbonScene", ImVec2( 0.f, float( currentTopPanelHeight_ ) * scaling - 1 ), ImGuiCond_Always );
    sceneSize_.x = std::round( std::min( sceneSize_.x, viewerRef.framebufferSize.x - 100 * scaling ) );
    sceneSize_.y = std::round( viewerRef.framebufferSize.y - float( currentTopPanelHeight_ - 2.0f ) * scaling );
    ImGui::SetWindowSize( "RibbonScene", sceneSize_, ImGuiCond_Always );
    ImGui::SetNextWindowSizeConstraints( ImVec2( 100 * scaling, -1.f ), ImVec2( viewerRef.framebufferSize.x / 2.f, -1.f ) ); // TODO take out limits to special place
    ImGui::PushStyleVar( ImGuiStyleVar_Alpha, 1.f );
    auto colorBg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    colorBg.w = 1.f;
    ImGui::PushStyleColor( ImGuiCol_WindowBg, colorBg );

    ImGui::Begin(
        "RibbonScene", nullptr,
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoResize
    );
    drawRibbonSceneListContent_( selectedObjs, allObj );
    drawRibbonSceneInformation_( selectedObjs );

    const auto newSize = drawRibbonSceneResizeLine_();// ImGui::GetWindowSize();
    static bool firstTime = true;
    bool manualSizeSet = false;
    if ( !firstTime && ( newSize.x != sceneSize_.x || newSize.y != sceneSize_.y ) )
    {
        manualSizeSet = true;
        sceneSize_ = newSize;
        fixViewportsSize_( viewerRef.framebufferSize.x, viewerRef.framebufferSize.y );
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    auto window = ImGui::FindWindowByName( "RibbonScene" );
    if ( !window || manualSizeSet )
        return;
    // this check is needed when resize of app window changes size of scene window
    auto lastWindowSize = window->Size;
    if ( !firstTime && lastWindowSize.x != sceneSize_.x )
    {
        sceneSize_.x = lastWindowSize.x;
        fixViewportsSize_( viewerRef.framebufferSize.x, viewerRef.framebufferSize.y );
    }
    if ( firstTime )
        firstTime = false;
}

void RibbonMenu::drawRibbonSceneListContent_( std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all )
{
    drawSceneListButtons_();
    ImGui::BeginChild( "Meshes", ImVec2( -1, -( informationHeight_ + transformHeight_ ) ), false );
    updateSceneWindowScrollIfNeeded_();
    auto children = SceneRoot::get().children();
    for ( const auto& child : children )
        draw_object_recurse_( *child, selected, all );
    makeDragDropTarget_( SceneRoot::get(), false, true, "" );

    // any click on empty space below Scene Tree removes object selection
    ImGui::BeginChild( "EmptySpace" );
    if ( ImGui::IsWindowHovered() && ImGui::IsMouseClicked( 0 ) )
    {
        for ( const auto& s : selected )
            if ( s )
                s->select( false );
    }
    ImGui::EndChild();

    ImGui::EndChild();
    sceneOpenCommands_.clear();

    reorderSceneIfNeeded_();
}

Vector2f RibbonMenu::drawRibbonSceneResizeLine_()
{
    auto size = sceneSize_;

    auto* window = ImGui::GetCurrentWindow();
    if ( !window )
        return size;

    auto scaling = menu_scaling();
    auto minX = 100.0f * scaling;
    auto maxX = getViewerInstance().framebufferSize.x * 0.5f;

    ImRect rectHover;
    ImRect rectDraw;
    rectHover.Min = ImGui::GetWindowPos();
    rectHover.Max = rectHover.Min;
    rectHover.Min.x += size.x - 3.5f * scaling;
    rectHover.Max.x += size.x + 3.5f * scaling;
    rectHover.Max.y += size.y;
    
    rectDraw = rectHover;
    rectDraw.Min.x += 1.5f * scaling;
    rectDraw.Max.x -= 1.5f * scaling;

    auto backupClipRect = window->ClipRect;
    window->ClipRect = rectHover;
    auto resizeId = window->GetID( "##resizePanel" );
    ImGui::ItemAdd( rectHover, resizeId, nullptr, ImGuiItemFlags_NoNav );
    bool hovered{ false }, held{ false };
    ImGui::ButtonBehavior( rectHover, resizeId, &hovered, &held,
        ImGuiButtonFlags_FlattenChildren | ImGuiButtonFlags_NoNavFocus );
    window->ClipRect = backupClipRect;

    if ( hovered || held )
    {
        ImGui::SetMouseCursor( ImGuiMouseCursor_ResizeEW );
        auto color = held ? ImGui::GetColorU32( ImGuiCol_ResizeGripActive ) : ImGui::GetColorU32( ImGuiCol_ResizeGripHovered );
        if ( held )
            size.x = std::clamp( ImGui::GetMousePos().x, minX, maxX );

        window->DrawList->PushClipRect( ImVec2( 0, 0 ), ImGui::GetMainViewport()->Size );
        window->DrawList->AddRectFilled( rectDraw.Min, rectDraw.Max, color );
        window->DrawList->PopClipRect();
    }
    return size;
}

void RibbonMenu::drawRibbonViewportsLabels_()
{
    const auto scaling = menu_scaling();
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::SemiBold ) );
    for ( const auto& vp : viewer->viewport_list )
    {
        constexpr std::array<const char*, 2> cProjModeString = { "Orthographic" , "Perspective" };
        std::string windowName = "##ProjectionMode" + std::to_string( vp.id.value() );
        std::string text;
        std::string label = vp.getParameters().label;
        if ( viewer->viewport_list.size() > 1 && label.empty() )
            label = fmt::format( "Viewport Id : {}", vp.id.value() );
        if ( !label.empty() )
            text = fmt::format( "{}, {}", label, cProjModeString[int( !vp.getParameters().orthographic )] );
        else
            text = fmt::format( "{}", cProjModeString[int( !vp.getParameters().orthographic )] );
        auto textSize = ImGui::CalcTextSize( text.c_str() );
        auto pos = viewer->viewportToScreen( Vector3f( width( vp.getViewportRect() ) - textSize.x - 25.0f * scaling,
            height( vp.getViewportRect() ) - textSize.y - 25.0f * scaling, 0.0f ), vp.id );
        ImGui::SetNextWindowPos( ImVec2( pos.x, pos.y ) );
        ImGui::Begin( windowName.c_str(), nullptr,
                      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize |
                      ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus );
        ImGui::Text( "%s", text.c_str() );
        ImGui::End();
    }
    ImGui::PopFont();
    ImGui::PopStyleVar();
}

void RibbonMenu::drawRibbonSceneInformation_( std::vector<std::shared_ptr<Object>>& /*selected*/ )
{
    const float newInfoHeight = std::ceil( drawSelectionInformation_() );

    const float newXfHeight = std::ceil( drawTransform_() );
    if ( newInfoHeight != informationHeight_ || newXfHeight != transformHeight_ )
    {
        informationHeight_ = newInfoHeight;
        transformHeight_ = newXfHeight;
        getViewerInstance().incrementForceRedrawFrames(1, true);
    }
}

void RibbonMenu::drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& selected )
{
    const auto selectedVisualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( ImGui::BeginPopupContextItem() )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, ImGui::GetStyle().WindowPadding );
        [[maybe_unused]] bool wasChanged = false, wasAction = false;
        if ( selectedVisualObjs.empty() )
        {
            wasChanged |= drawGeneralOptions_( selected );
            wasAction |= drawRemoveButton_( selected );
            wasAction |= drawGroupUngroupButton_( selected );
            wasAction |= drawCloneButton_( selected );
        }
        else if ( ImGui::BeginTable( "##DrawOptions", 2, ImGuiTableFlags_BordersInnerV ) )
        {
            ImGui::TableNextColumn();
            wasChanged |= drawGeneralOptions_( selected );
            wasChanged |= drawDrawOptionsCheckboxes_( selectedVisualObjs );
            wasChanged |= drawAdvancedOptions_( selectedVisualObjs );
            ImGui::TableNextColumn();
            wasChanged |= drawDrawOptionsColors_( selectedVisualObjs );
            wasAction |= drawRemoveButton_( selected );
            wasAction |= drawGroupUngroupButton_( selected );
            wasAction |= drawCloneButton_( selected );
            wasAction |= drawCloneSelectionButton_( selected );
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
        //uncomment to close context menu on any change
        //if ( wasChanged || wasAction )
        if ( wasAction )
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

bool RibbonMenu::drawTransformContextMenu_( const std::shared_ptr<Object>& selected )
{
    if ( !ImGui::BeginPopupContextItem( "TransformContextWindow" ) )
        return false;

    auto buttonSize = 100.0f * menu_scaling();

    struct Transform
    {
        AffineXf3f xf;
        bool uniformScale{ true };
    };
    auto serializeTransform = [] ( Json::Value& root, const Transform& tr )
    {
        root["Name"] = "MeshLib Transform";
        serializeToJson( tr.xf, root["XF"] );
        root["UniformScale"] = tr.uniformScale;
    };
    auto deserializeTransform = [] ( const Json::Value& root ) -> std::optional<Transform>
    {
        if ( !root.isObject() || root["Name"].asString() != "MeshLib Transform" )
            return std::nullopt;

        AffineXf3f xf;
        deserializeFromJson( root["XF"], xf );
        auto uniformScale = root["UniformScale"].asBool();
        return Transform{ xf, uniformScale };
    };

    const auto& startXf = selected->xf();
#if !defined( __EMSCRIPTEN__ )
    if ( UI::button( "Copy", Vector2f( buttonSize, 0 ) ) )
    {
        Json::Value root;
        serializeTransform( root, { startXf, uniformScale_ } );
        transformClipboardText_ = root.toStyledString();
        SetClipboardText( transformClipboardText_ );
        ImGui::CloseCurrentPopup();
    }
#endif
    if ( ImGui::IsWindowAppearing() )
        transformClipboardText_ = GetClipboardText();

    if ( !transformClipboardText_.empty() )
    {
        Json::Value root;
        Json::CharReaderBuilder readerBuilder;
        std::unique_ptr<Json::CharReader> reader{ readerBuilder.newCharReader() };
        std::string error;
        if ( reader->parse( transformClipboardText_.data(), transformClipboardText_.data() + transformClipboardText_.size(), &root, &error ) )
        {
            if ( auto tr = deserializeTransform( root ))
            {
                if ( UI::button( "Paste", Vector2f( buttonSize, 0 ) ) )
                {
                    AppendHistory<ChangeXfAction>( "Change XF", selected );
                    selected->setXf( tr->xf );
                    uniformScale_ = tr->uniformScale;
                    ImGui::CloseCurrentPopup();
                }
            }
        }
    }

    if ( UI::button( "Save to file", Vector2f( buttonSize, 0 ) ) )
    {
        auto filename = saveFileDialog( { "Transform", {}, { {"JSON (.json)", "*.json"} } } );
        if ( !filename.empty() )
        {
            Json::Value root;
            serializeTransform( root, { startXf, uniformScale_ } );

            std::ofstream ofs( filename );
            if ( ofs )
                ofs << root.toStyledString();
            else
                spdlog::error( "Cannot open file for writing" );
        }
        ImGui::CloseCurrentPopup();
    }

    if ( UI::button( "Load from file", Vector2f( buttonSize, 0 ) ) )
    {
        auto filename = openFileDialog( { "", {}, { {"JSON (.json)", "*.json"} } } );
        if ( !filename.empty() )
        {
            std::ifstream ifs( filename );
            if ( ifs )
            {
                std::string text( ( std::istreambuf_iterator<char>( ifs ) ), std::istreambuf_iterator<char>() );

                Json::Value root;
                Json::CharReaderBuilder readerBuilder;
                std::unique_ptr<Json::CharReader> reader{ readerBuilder.newCharReader() };
                std::string error;
                if ( reader->parse( text.data(), text.data() + text.size(), &root, &error ) )
                {
                    if ( auto tr = deserializeTransform( root ))
                    {
                        AppendHistory<ChangeXfAction>( "Change XF", selected );
                        selected->setXf( tr->xf );
                        uniformScale_ = tr->uniformScale;
                    } else
                    {
                        spdlog::error( "Cannot parse transform" );
                    }
                }
                else
                {
                    spdlog::error( "Cannot parse transform" );
                }
            }
            else
            {
                spdlog::error( "Cannot open file for reading" );
            }
        }
        ImGui::CloseCurrentPopup();
    }

    if ( startXf != AffineXf3f() )
    {
        auto item = RibbonSchemaHolder::schema().items.find( "Apply Transform" );
        if ( item != RibbonSchemaHolder::schema().items.end() &&
            item->second.item->isAvailable( selectedObjectsCache_ ).empty() &&
            UI::button( "Apply", Vector2f( buttonSize, 0 ) ) )
        {
            item->second.item->action();
            ImGui::CloseCurrentPopup();
        }
        UI::setTooltipIfHovered( "Transforms object and resets transform value to identity.", menu_scaling() );

        if ( UI::button( "Reset", Vector2f( buttonSize, 0 ) ) )
        {
            AppendHistory<ChangeXfAction>( "Reset XF", selected );
            selected->setXf( AffineXf3f() );
            ImGui::CloseCurrentPopup();
        }
        UI::setTooltipIfHovered( "Resets transform value to identity.", menu_scaling() );
    }
    ImGui::EndPopup();
    return true;
}

const char* RibbonMenu::getSceneItemIconByTypeName_( const std::string& typeName ) const
{
    if ( typeName == ObjectMesh::TypeName() )
        return "\xef\x82\xac";
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
    if ( typeName == ObjectVoxels::TypeName() )
        return "\xef\x86\xb3";
#endif
    if ( typeName == ObjectPoints::TypeName() )
        return "\xef\x84\x90";
    if ( typeName == ObjectLines::TypeName() )
        return "\xef\x87\xa0";
    if ( typeName == ObjectDistanceMap::TypeName() )
        return "\xef\xa1\x8c";
    if ( typeName == ObjectLabel::TypeName() )
        return "\xef\x81\xb5";
    return "\xef\x88\xad";
}

void RibbonMenu::drawCustomObjectPrefixInScene_( const Object& obj )
{
    auto imageSize = ImGui::GetFrameHeight();
    auto* imageIcon = RibbonIcons::findByName( obj.typeName(), imageSize,
                                               RibbonIcons::ColorType::White,
                                               RibbonIcons::IconType::ObjectTypeIcon );

    if ( !imageIcon )
    {
        auto font = fontManager_.getFontByType( RibbonFontManager::FontType::Icons );
        font->Scale = fontManager_.getFontSizeByType( RibbonFontManager::FontType::Default ) /
            fontManager_.getFontSizeByType( RibbonFontManager::FontType::Icons );
        ImGui::PushFont( font );

        ImGui::Text( "%s", getSceneItemIconByTypeName_( obj.typeName() ) );

        ImGui::PopFont();
        font->Scale = 1.0f;
    }
    else
    {
        auto multColor = ImGui::GetStyleColorVec4( ImGuiCol_Text );
        ImGui::Image( *imageIcon, ImVec2( imageSize, imageSize ), multColor );
    }
    ImGui::SameLine();
}

void RibbonMenu::addRibbonItemShortcut_( const std::string& itemName, const ShortcutManager::ShortcutKey& key, ShortcutManager::Category category )
{
    auto itemIt = RibbonSchemaHolder::schema().items.find( itemName );
    if ( itemIt != RibbonSchemaHolder::schema().items.end() )
    {
        auto caption = itemIt->second.caption.empty() ? itemIt->first : itemIt->second.caption;
        shortcutManager_->setShortcut( key, { category, caption,[item = itemIt->second.item, this]()
        {
            itemPressed_( item, getRequirements_( item ).empty() );
        } } );
    }
#ifndef __EMSCRIPTEN__
    else
        assert( !"item not found" );
#endif
}

void RibbonMenu::setupShortcuts_()
{
    if ( !shortcutManager_ )
        shortcutManager_ = std::make_shared<ShortcutManager>();

    shortcutManager_->setShortcut( { GLFW_KEY_H,0 }, { ShortcutManager::Category::View, "Toggle selected objects visibility", [] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
        bool atLeastOne = false;
        for ( const auto& data : selected )
            if ( data )
                if ( data->isVisible( viewportid ) )
                {
                    atLeastOne = true;
                    break;
                }
        for ( const auto& data : selected )
            if ( data )
                data->setVisible( !atLeastOne, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_F1,0 }, { ShortcutManager::Category::Info, "Show this help with hot keys",[this] ()
   {
       showShortcuts_ = !showShortcuts_;
   } } );
    shortcutManager_->setShortcut( { GLFW_KEY_D,0 }, { ShortcutManager::Category::Info, "Toggle statistics window",[this] ()
    {
        showStatistics_ = !showStatistics_;
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_F,0 }, { ShortcutManager::Category::View, "Toggle shading of selected objects",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( MeshVisualizePropertyType::FlatShading, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_I,0 }, { ShortcutManager::Category::View, "Invert normals of selected objects",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( VisualizeMaskType::InvertedNormals, viewportid );
    } }  );
    shortcutManager_->setShortcut( { GLFW_KEY_L,0 }, { ShortcutManager::Category::View, "Toggle edges on selected meshes",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
                sel->toggleVisualizeProperty( MeshVisualizePropertyType::Edges, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_O,0 }, { ShortcutManager::Category::View, "Toggle orthographic in current viewport",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        viewport.setOrthographic( !viewport.getParameters().orthographic );
    } }  );
    shortcutManager_->setShortcut( { GLFW_KEY_T,0 }, { ShortcutManager::Category::View, "Toggle faces on selected meshes",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( MeshVisualizePropertyType::Faces, viewportid );
    } }  );
    shortcutManager_->setShortcut( { GLFW_KEY_DOWN,0 }, { ShortcutManager::Category::Objects, "Select next object",[] ()
    {
        changeSelection( true,0 );
    } }  );
    shortcutManager_->setShortcut( { GLFW_KEY_DOWN,GLFW_MOD_SHIFT }, { ShortcutManager::Category::Objects, "Add next object to selection",[] ()
    {
        changeSelection( true,GLFW_MOD_SHIFT );
    } }  );
    shortcutManager_->setShortcut( { GLFW_KEY_UP,0 }, { ShortcutManager::Category::Objects, "Select previous object",[] ()
    {
        changeSelection( false,0 );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_UP,GLFW_MOD_SHIFT }, { ShortcutManager::Category::Objects, "Add previous object to selection",[] ()
    {
        changeSelection( false,GLFW_MOD_SHIFT );
    } }  );

    addRibbonItemShortcut_( "Ribbon Scene Select all", { GLFW_KEY_A, GLFW_MOD_CONTROL }, ShortcutManager::Category::Objects );
    addRibbonItemShortcut_( "Fit data", { GLFW_KEY_F, GLFW_MOD_CONTROL }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Select objects", { GLFW_KEY_Q, GLFW_MOD_CONTROL }, ShortcutManager::Category::Objects );
    addRibbonItemShortcut_( "Open files", { GLFW_KEY_O, GLFW_MOD_CONTROL }, ShortcutManager::Category::Scene );
    addRibbonItemShortcut_( "Save Scene", { GLFW_KEY_S, GLFW_MOD_CONTROL }, ShortcutManager::Category::Scene );
    addRibbonItemShortcut_( "Save Scene As", { GLFW_KEY_S, GLFW_MOD_CONTROL | GLFW_MOD_SHIFT }, ShortcutManager::Category::Scene );
    addRibbonItemShortcut_( "New", { GLFW_KEY_N, GLFW_MOD_CONTROL }, ShortcutManager::Category::Scene );
    addRibbonItemShortcut_( "Ribbon Scene Show only previous", { GLFW_KEY_F3, 0 }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Ribbon Scene Show only next", { GLFW_KEY_F4, 0 }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Ribbon Scene Rename", { GLFW_KEY_F2, 0 }, ShortcutManager::Category::Objects );
    addRibbonItemShortcut_( "Ribbon Scene Remove selected objects", { GLFW_KEY_R, GLFW_MOD_SHIFT }, ShortcutManager::Category::Objects );
}

void RibbonMenu::drawShortcutsWindow_()
{
    const auto& style = ImGui::GetStyle();
    const auto scaling = menu_scaling();
    float windowWidth = 920.0f * scaling;

    const auto& shortcutList = shortcutManager_->getShortcutList();
    // header size
    float windowHeight = ( 6 * cDefaultItemSpacing + fontManager_.getFontSizeByType( RibbonFontManager::FontType::Headline ) + style.CellPadding.y + StyleConsts::Modal::bigTitlePadding ) * scaling;
    // find max column size
    int leftNumCategories = 0;
    int rightNumCategories = 0;
    int leftNumKeys = 0;
    int rightNumKeys = 0;
    auto lastCategory = ShortcutManager::Category::Count;// invalid for first one
    for ( int i = 0; i < shortcutList.size(); ++i )
    {
        const auto& [key, category, text] = shortcutList[i];
        bool right = int( category ) >= int( ShortcutManager::Category::Count ) / 2;
        auto& catCounter = right ? rightNumCategories : leftNumCategories;
        auto& keyCounter = right ? rightNumKeys : leftNumKeys;
        if ( category != lastCategory )
        {
            ++catCounter;
            lastCategory = category;
        }
        ++keyCounter;
    }
    auto leftSize = ( leftNumCategories * ( fontManager_.getFontSizeByType( RibbonFontManager::FontType::BigSemiBold ) + cSeparateBlocksSpacing + 2 * style.CellPadding.y ) +
        leftNumKeys * ( fontManager_.getFontSizeByType( RibbonFontManager::FontType::Default ) + cButtonPadding + 2 * cDefaultItemSpacing ) ) * scaling;
    auto rightSize = ( rightNumCategories * ( fontManager_.getFontSizeByType( RibbonFontManager::FontType::BigSemiBold ) + cSeparateBlocksSpacing + 2 * style.CellPadding.y ) +
        rightNumKeys * ( fontManager_.getFontSizeByType( RibbonFontManager::FontType::Default ) + cButtonPadding + 2 * cDefaultItemSpacing ) ) * scaling;
    // calc window size for better positioning
    windowHeight += std::max( leftSize, rightSize );

    const float minHeight = 200.0f * scaling;
    windowHeight = std::clamp( windowHeight, minHeight, float( getViewerInstance().framebufferSize.y ) - 100.0f * scaling );

    ImVec2 windowPos;
    windowPos.x = ( getViewerInstance().framebufferSize.x - windowWidth ) * 0.5f;
    windowPos.y = ( getViewerInstance().framebufferSize.y - windowHeight ) * 0.5f;

    ImGui::SetNextWindowPos( windowPos, ImGuiCond_Appearing );
    ImGui::SetNextWindowSize( ImVec2( windowWidth, windowHeight ), ImGuiCond_Always );

    if ( !ImGui::IsPopupOpen( "HotKeys" ) )
        ImGui::OpenPopup( "HotKeys" );

    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( StyleConsts::Modal::bigTitlePadding * scaling, 0.0f ) );
    if ( !ImGui::BeginModalNoAnimation( "HotKeys", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        ImGui::PopStyleVar();
        return;
    }
    ImGui::PopStyleVar();

    ImGui::SetCursorPosY( StyleConsts::Modal::bigTitlePadding * scaling );
    if ( ImGui::ModalBigTitle( "Hotkeys", scaling ) )
    {
        ImGui::CloseCurrentPopup();
        showShortcuts_ = false;
        ImGui::EndPopup();
        return;
    }

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { cDefaultItemSpacing * scaling, 2 * cDefaultItemSpacing * scaling } );

    int lineIndexer = 0;
    auto addReadOnlyLine = [scaling, &style, &lineIndexer] ( const std::string& line )
    {
        const auto textWidth = ImGui::CalcTextSize( line.c_str() ).x;
        // read only so const_sast should be ok
        auto itemWidth = std::max( textWidth + 2 * style.FramePadding.x, 30.0f * scaling );
        ImGui::PushItemWidth( itemWidth );

        auto framePaddingX = std::max( style.FramePadding.x, ( itemWidth - textWidth ) / 2.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { framePaddingX, cButtonPadding * scaling } );
        ImGui::InputText( ( "##" + line + std::to_string( ++lineIndexer ) ).c_str(), const_cast< std::string& >( line ), ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_AutoSelectAll );
        ImGui::PopItemWidth();
        ImGui::PopStyleVar();
    };
    
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + cDefaultItemSpacing * scaling );

    ImGui::BeginChild( "##Hotkeys_table_chalid" );
    if ( ImGui::BeginTable( "HotKeysTable", 2, ImGuiTableFlags_SizingStretchSame ) )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { cButtonPadding * scaling, style.FramePadding.y } );
        ImGui::TableNextColumn();
        bool secondColumnStarted = false;
        lastCategory = ShortcutManager::Category::Count;// invalid for first one
        for ( int i = 0; i < shortcutList.size(); ++i )
        {
            const auto& [key, category, text] = shortcutList[i];

            if ( !secondColumnStarted && int( category ) >= int( ShortcutManager::Category::Count ) / 2 )
            {
                // start second column
                ImGui::TableNextColumn();
                ImGui::Indent();
                secondColumnStarted = true;
            }

            if ( category != lastCategory )
            {
                // draw category line
                ImGui::PushFont( fontManager_.getFontByType( MR::RibbonFontManager::FontType::BigSemiBold ) );
                UI::separator( scaling, ShortcutManager::categoryNames[int( category )].c_str() );
                ImGui::PopFont();
                lastCategory = category;
            }
            // draw hotkey
            auto transparentColor = ImGui::GetStyleColorVec4( ImGuiCol_Text );
            transparentColor.w *= 0.5f;
            ImGui::PushStyleColor( ImGuiCol_Text, transparentColor );
            ImGui::Text( "%s", text.c_str() );
            ImGui::PopStyleColor();

            float textSize = ImGui::CalcTextSize( text.c_str() ).x;
            ImGui::SameLine( 0, 260 * scaling - textSize );

            if ( key.mod & GLFW_MOD_CONTROL )
            {
                ImGui::SetCursorPosY( ImGui::GetCursorPosY() - cButtonPadding * scaling );
                addReadOnlyLine( ShortcutManager::getModifierString( GLFW_MOD_CONTROL ) );
                ImGui::SameLine( 0, style.ItemInnerSpacing.x );
                ImGui::SetCursorPosY( ImGui::GetCursorPosY() - cButtonPadding * scaling );
                ImGui::Text( "+" );
                ImGui::SameLine( 0, style.ItemInnerSpacing.x );
            }

            if ( key.mod & GLFW_MOD_ALT )
            {
                ImGui::SetCursorPosY( ImGui::GetCursorPosY() - cButtonPadding * scaling );
                addReadOnlyLine( ShortcutManager::getModifierString( GLFW_MOD_ALT ) );
                ImGui::SameLine( 0, style.ItemInnerSpacing.x );
                ImGui::SetCursorPosY( ImGui::GetCursorPosY() - cButtonPadding * scaling );
                ImGui::Text( "+" );
                ImGui::SameLine( 0, style.ItemInnerSpacing.x );
            }

            if ( key.mod & GLFW_MOD_SHIFT )
            {
                ImGui::SetCursorPosY( ImGui::GetCursorPosY() - cButtonPadding * scaling );
                addReadOnlyLine( ShortcutManager::getModifierString( GLFW_MOD_SHIFT ) );
                ImGui::SameLine( 0, style.ItemInnerSpacing.x );
                ImGui::SetCursorPosY( ImGui::GetCursorPosY() - cButtonPadding * scaling );
                ImGui::Text( "+" );
                ImGui::SameLine( 0, style.ItemInnerSpacing.x );
            }

            std::string keyStr = ShortcutManager::getKeyString( key.key );
            bool isArrow = key.key == GLFW_KEY_UP || key.key == GLFW_KEY_DOWN || key.key == GLFW_KEY_LEFT || key.key == GLFW_KEY_RIGHT;
            ImFont* font = nullptr;
            if ( isArrow )
            {
                font = fontManager_.getFontByType( RibbonFontManager::FontType::Icons );
                font->Scale = cDefaultFontSize / cBigIconSize;
                ImGui::PushFont( font );
            }

            ImGui::SetCursorPosY( ImGui::GetCursorPosY() - cButtonPadding * scaling );
            addReadOnlyLine( keyStr );

            if ( isArrow )
            {
                ImGui::PopFont();
            }
        }

        ImGui::PopStyleVar();
        ImGui::EndTable();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::EndPopup();
}

void RibbonMenu::beginTopPanel_()
{
    const auto scaling = menu_scaling();
    ImGui::SetNextWindowPos( ImVec2( 0, 0 ) );
    ImGui::SetNextWindowSize( ImVec2( ( float ) Viewer::instanceRef().framebufferSize.x, currentTopPanelHeight_ * scaling ) );

    ImGui::PushStyleVar( ImGuiStyleVar_Alpha, 1.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, 5.0f * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_ChildRounding, 5.0f * scaling );
    auto colorBg = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TopPanelBackground );
    if ( collapseState_ != CollapseState::Opened )
        colorBg.a = 255;
    else
    {
        colorBg.a = 228;
        ImGui::GetBackgroundDrawList()->AddRectFilled( ImVec2( 0.0f, 0.0f ),
            ImVec2( sceneSize_.x, currentTopPanelHeight_ * scaling ),
            ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::Background ).getUInt32() );
    }
    ImGui::PushStyleColor( ImGuiCol_WindowBg, colorBg.getUInt32() );

    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0, 0 ) );
    ImGui::Begin(
        "TopPanel", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
    );
    ImGui::PopStyleVar();
    // for all items
    ProgressBar::setup( scaling );
}

void RibbonMenu::endTopPanel_()
{
    ImGui::Dummy( ImVec2( 0, 0 ) );
    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar( 3 );
}

void RibbonMenu::drawTopPanel_()
{
    switch ( collapseState_ )
    {
        case MR::RibbonMenu::CollapseState::Closed:
            if ( currentTopPanelHeight_ != topPanelHiddenHeight_ )
            {
                currentTopPanelHeight_ = topPanelHiddenHeight_;
            }
            break;
        default:
            if ( currentTopPanelHeight_ != topPanelOpenedHeight_ )
            {
                currentTopPanelHeight_ = topPanelOpenedHeight_;
            }
            break;
    }
    drawTopPanelOpened_();
}

void RibbonMenu::drawTopPanelOpened_()
{
    beginTopPanel_();

    const auto& style = ImGui::GetStyle();
    auto itemSpacing = style.ItemSpacing;
    itemSpacing.x = cRibbonItemInterval * menu_scaling();
    auto cellPadding = style.CellPadding;
    cellPadding.x = itemSpacing.x;
    auto framePadding = style.FramePadding;
    framePadding.x = 0.0f;

    drawHeaderPannel_();

    // tab content position
    ImGui::SetCursorPosX( style.CellPadding.x * 2 );
    ImGui::SetCursorPosY( ( cTabYOffset + cTabHeight ) * menu_scaling() + 2 );

    ImGuiTableFlags tableFlags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerV;

    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::Small ) );
    const auto& tab = RibbonSchemaHolder::schema().tabsOrder[activeTabIndex_].name;
    if ( collapseState_ != CollapseState::Closed )
    {
        auto tabIt = RibbonSchemaHolder::schema().tabsMap.find( tab );
        if ( tabIt != RibbonSchemaHolder::schema().tabsMap.end() )
        {
            ImGui::PushStyleColor( ImGuiCol_TableBorderLight, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Borders ).getUInt32() );
            ImGui::PushStyleColor( ImGuiCol_ScrollbarBg, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TopPanelBackground ).getUInt32() );
            ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, cellPadding );
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );
            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, framePadding ); // frame padding cause horizontal scrollbar which is not needed
            ImGui::PushStyleVar( ImGuiStyleVar_ScrollbarSize, cScrollBarSize * menu_scaling() );
            if ( ImGui::BeginTable( ( tab + "##table" ).c_str(), int( tabIt->second.size() + 1 ), tableFlags ) )
            {
                setupItemsGroup_( tabIt->second, tab );
                auto config = setupItemsGroupConfig_( tabIt->second, tab );
                ImGui::TableNextRow();
                for ( int i = 0; i < tabIt->second.size(); ++i )
                {
                    const auto& group = tabIt->second[i];
                    ImGui::TableNextColumn();
                    drawItemsGroup_( tab, group, config[i] );
                }
                ImGui::TableNextColumn(); // fictive
                ImGui::EndTable();
            }
            ImGui::PopStyleVar( 4 );
            ImGui::PopStyleColor( 2 );
        }
    }
    ImGui::PopFont();
    endTopPanel_();
}

void RibbonMenu::fixViewportsSize_( int width, int height )
{
    auto viewportsBounds = viewer->getViewportsBounds();
    auto minMaxDiff = viewportsBounds.max - viewportsBounds.min;

    const float topPanelHeightScaled =
        ( collapseState_ == CollapseState::Pinned ? topPanelOpenedHeight_ : topPanelHiddenHeight_ ) *
        menu_scaling();
    for ( auto& vp : viewer->viewport_list )
    {
        auto rect = vp.getViewportRect();

        const float sceneWidth = sceneSize_.x;

        auto widthRect = MR::width( rect );
        auto heightRect = MR::height( rect );

        rect.min.x = ( rect.min.x - viewportsBounds.min.x ) / minMaxDiff.x * ( width - sceneWidth ) + sceneWidth;
        rect.min.y = ( rect.min.y - viewportsBounds.min.y ) / minMaxDiff.y * ( height - ( topPanelHeightScaled - 2 ) ); // -2 - buffer pixel
        rect.max.x = rect.min.x + widthRect / minMaxDiff.x * ( width - sceneWidth );
        rect.max.y = rect.min.y + heightRect / minMaxDiff.y * ( height - ( topPanelHeightScaled - 2 ) ); // -2 - buffer pixel
        if ( MR::width( rect ) <= 0 || MR::height( rect ) <= 0 )
            continue;
        vp.setViewportRect( rect );
    }
}

bool RibbonMenu::drawCollapsingHeader_( const char* label, ImGuiTreeNodeFlags flags )
{
    return RibbonButtonDrawer::CustomCollapsingHeader( label, flags );
}

void RibbonMenu::highlightBlocking_()
{
    if ( blockingHighlightTimer_ <= 0.0f )
        return;
    if ( !activeBlockingItem_.item )
    {
        blockingHighlightTimer_ = 0.0f;
        return;
    }
    auto pluginWindowName = activeBlockingItem_.item->uiName();
    auto* window = ImGui::FindWindowByName( pluginWindowName.c_str() );
    if ( !window || blockingHighlightTimer_ <= 0.0f )
    {
        blockingHighlightTimer_ = 0.0f;
        return;
    }
    auto scaling = menu_scaling();
    if ( int( blockingHighlightTimer_ / 0.2f ) % 2 == 1 )
    {
        Color highlightColor = Color( 255, 161, 13, 255 );
        ImGui::FocusWindow( window );
        auto drawList = window->DrawList;
        drawList->PushClipRect( ImVec2( 0, 0 ), ImGui::GetIO().DisplaySize );
        drawList->AddRect(
            ImVec2( window->Pos.x - 2.0f * scaling, window->Pos.y - 2.0f * scaling ),
            ImVec2( window->Pos.x + window->Size.x + 2.0f * scaling, window->Pos.y + window->Size.y + 2.0f * scaling ),
            highlightColor.getUInt32(), 0.0f, 0, 2.0f * scaling );
        drawList->PopClipRect();
    }
    getViewerInstance().incrementForceRedrawFrames();
    blockingHighlightTimer_ -= ImGui::GetIO().DeltaTime;
}

}
