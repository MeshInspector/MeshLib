#include "MRRibbonMenu.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRProgressBar.h"
#include "MRColorTheme.h"
#include "MRAppendHistory.h"
#include "MRCommandLoop.h"
#include "MRMesh/MRChangeSceneObjectsOrder.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRSpdlog.h"
#include "MRRibbonIcons.h"
#include "MRRibbonConstants.h"
#include "ImGuiHelpers.h"
#include "MRMesh/MRString.h"
#include "MRImGuiImage.h"
#include "ImGuiHelpers.h"
#include <imgui_internal.h> // needed here to fix items dialogs windows positions
#include <misc/freetype/imgui_freetype.h> // for proper font loading

#if defined(__APPLE__) && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-volatile"
#endif

#include <GLFW/glfw3.h>
#include "MRMesh/MRObjectLabel.h"

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

RibbonMenu::RibbonMenu()
{
#ifndef __EMSCRIPTEN__
    timerThread_ = std::thread( [this] ()
    {
        MR::SetCurrentThreadName( "RibbonMenu timer thread" );
        while ( asyncTimer_.waitBlocking() != AsyncTimer::Event::Terminate )
        {
            CommandLoop::appendCommand( [] ()
            {
                getViewerInstance().incrementForceRedrawFrames();
            } );
        }
    } );
#endif
}

RibbonMenu::~RibbonMenu()
{
    asyncTimer_.terminate();
#ifndef __EMSCRIPTEN__
    timerThread_.join();
#endif
}

void RibbonMenu::init( MR::Viewer* _viewer )
{
    Menu::init( _viewer );
    readMenuItemsStructure_();

    RibbonIcons::load();

    callback_draw_viewer_window = [] ()
    {};

    // Draw additional windows
    callback_draw_custom_window = [&] ()
    {
        selectedObjectsCache_ = getAllObjectsInTree<const Object>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        drawTopPanel_();

        drawActiveBlockingDialog_();
        drawActiveNonBlockingDialogs_();

        drawQuickAccessMenu_();
        drawRibbonSceneList_();
        drawRibbonViewportsLabels_();

        draw_helpers();
    };

    buttonDrawer_.setMenu( this );
    buttonDrawer_.setFontMenager( &fontManager_ );
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
}

void RibbonMenu::shutdown()
{
    for ( auto& item : RibbonSchemaHolder::schema().items )
    {
        if ( item.second.item && item.second.item->isActive() )
            item.second.item->action();
    }
    Menu::shutdown();
    RibbonIcons::free();
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
    fixViewportsSize_( Viewer::instanceRef().window_width, Viewer::instanceRef().window_height );
}

bool RibbonMenu::isTopPannelPinned() const
{
    return collapseState_ == CollapseState::Pinned;
}

void RibbonMenu::readQuickAccessList( const Json::Value& root )
{
    RibbonSchemaLoader::readMenuItemsList( root, quickAccessList_ );
}

void RibbonMenu::resetQuickAccessList()
{
    quickAccessList_ = RibbonSchemaHolder::schema().defaultQuickAccessList;
}

void RibbonMenu::setSceneSize( const Vector2i& size )
{
    sceneSize_ = ImVec2( float( size.x ), float( size.y ) );
    auto& viewerRef = Viewer::instanceRef();
    fixViewportsSize_( viewerRef.window_width, viewerRef.window_height );
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
            ImRect rectOuter = ImGui::GetWindowAllowedExtentRect( menuWindow );
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
            searchResult_ = search_();

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
                const auto& tabName = RibbonSchemaHolder::schema().tabsOrder[foundItem.tabIndex];
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
            fixViewportsSize_( Viewer::instanceRef().window_width, Viewer::instanceRef().window_height );
            openedTimer_ = openedMaxSecs_;
            asyncTimer_.resetTime();
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
            fixViewportsSize_( Viewer::instanceRef().window_width, Viewer::instanceRef().window_height );
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
            ImGuiHoveredFlags_AllowWhenBlockedByActiveItem |
            ImGuiHoveredFlags_AllowWhenDisabled );
        if ( hovered && openedTimer_ <= openedMaxSecs_ )
        {
            asyncTimer_.resetTime();
            openedTimer_ = openedMaxSecs_;
            collapseState_ = CollapseState::Opened;
        }
        else
        {
            openedTimer_ -= ImGui::GetIO().DeltaTime;
            asyncTimer_.setTimeIfNotSet( std::chrono::system_clock::now() + std::chrono::milliseconds( std::llround( openedTimer_ * 1000 ) ) );
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
    const auto availableWidth = getViewerInstance().window_width;
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
        ImVec2( float( getViewerInstance().window_width ), ( cTabHeight + cTabYOffset ) * menuScaling ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::HeaderBackground ).getUInt32() );

    drawHeaderQuickAccess_();

    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::SemiBold ) );
    // TODO_store: this needs recalc only on scaling change, no need to calc each frame
    std::vector<float> textSizes( RibbonSchemaHolder::schema().tabsOrder.size() );// TODO_store: add to some store at the beginning not to calc each time  
    std::vector<float> tabSizes( RibbonSchemaHolder::schema().tabsOrder.size() );// TODO_store: add to some store at the beginning not to calc each time  
    auto summaryTabPannelSize = 2 * 12.0f * menuScaling - cTabsInterval * menuScaling; // init shift (by eye, not defined in current design maket)
    for ( int i = 0; i < tabSizes.size(); ++i )
    {
        const auto& tabStr = RibbonSchemaHolder::schema().tabsOrder[i];
        textSizes[i] = ImGui::CalcTextSize( tabStr.c_str() ).x;
        tabSizes[i] = std::max( textSizes[i] + cTabLabelMinPadding * 2 * menuScaling, cTabMinimumWidth * menuScaling );
        summaryTabPannelSize += ( tabSizes[i] + cTabsInterval * menuScaling );
    }
    // prepare active button
    bool hasActive = activeBlockingItem_.item || !activeNonBlockingItems_.empty();
    float activeTextSize = ImGui::CalcTextSize( "Active" ).x;
    float activeBtnSize = std::max( activeTextSize + cTabLabelMinPadding * 2 * menuScaling, cTabMinimumWidth * menuScaling );
    if ( hasActive )
        summaryTabPannelSize += ( activeBtnSize + cTabsInterval * menuScaling );

    // 40 - search button size (by eye)
    // 40 - collapse button size (by eye)
    auto availWidth = ImGui::GetContentRegionAvail().x - 2 * 40.0f * menuScaling; 

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
        if ( buttonDrawer_.drawCustomStyledButton( "\xef\x81\x88", ImVec2( cTopPanelScrollBtnSize * menuScaling, ( cTabYOffset + cTabHeight ) * menuScaling ), btnSize ) )
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
        const auto& tabStr = RibbonSchemaHolder::schema().tabsOrder[i];
        const auto& tabWidth = tabSizes[i];
        ImVec2 tabBbMaxPoint( basePos.x + tabWidth, basePos.y + cTabHeight * menuScaling + 2 ); // +2 due to TabItemBackground internal offset
        ImRect tabRect( basePos, tabBbMaxPoint );
        std::string strId = "##" + tabStr + "TabId"; // TODO_store: add to some store at the beginning not to calc each time
        auto tabId = window->GetID( strId.c_str() );
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
        ImGui::SetCursorPosY( 2 * cTabYOffset * menuScaling );

        if ( activeTabIndex_ == i )
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActiveText ).getUInt32() );
        else
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText ).getUInt32() );
        ImGui::RenderText( ImGui::GetCursorPos(), tabStr.c_str(), tabStr.c_str() + tabStr.size(), false );
        ImGui::PopStyleColor();

        basePos.x += ( tabWidth + cTabsInterval * menuScaling );
    }
    if ( hasActive )
    {
        drawActiveListButton_( basePos, activeBtnSize, activeTextSize );
        basePos.x += ( activeBtnSize + cTabsInterval * menuScaling );
    }
    ImGui::EndChild();
    if ( needFwdBtn )
    {
        ImGui::SameLine();
        ImGui::SetCursorPosX( ImGui::GetCursorPosX() + cTabsInterval * menuScaling );
        const float btnSize = 0.5f * fontManager_.getFontSizeByType( RibbonFontManager::FontType::Icons );
        if ( buttonDrawer_.drawCustomStyledButton( "\xef\x81\x91", ImVec2( cTopPanelScrollBtnSize * menuScaling, ( cTabYOffset + cTabHeight ) * menuScaling ), btnSize ) )
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
    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddLine( ImVec2( 0, separateLinePos ), ImVec2( float( getViewerInstance().window_width ), separateLinePos ),
                                                                  ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::HeaderSeparator ).getUInt32() );


    ImGui::SetCursorPos( ImVec2( float( getViewerInstance().window_width ) - 70.0f * menuScaling, cTabYOffset* menuScaling ) );
    drawSearchButton_();

    ImGui::SetCursorPos( ImVec2( float( getViewerInstance().window_width ) - 30.0f * menuScaling, cTabYOffset * menuScaling ) );
    drawCollapseButton_();
}

void RibbonMenu::drawActiveListButton_( const ImVec2& basePos, float btnSize, float textSize )
{
    auto scaling = menu_scaling();
    auto windowPos = ImGui::GetCurrentContext()->CurrentWindow->Pos;
    auto xPos = basePos.x + cTabsInterval * scaling;
    ImGui::SetCursorPosX( xPos - windowPos.x );
    ImGui::SetNextItemWidth( btnSize );

    bool closeBlocking = false;
    std::vector<bool> closeNonBlocking( activeNonBlockingItems_.size(), false );

    ImGui::SetCursorPosY( cTabYOffset * scaling );

    float fontSize = ImGui::GetFontSize();
    auto framePaddingY = ( cTabHeight * scaling - fontSize ) * 0.5f;
    auto framePadding = ImGui::GetStyle().FramePadding;
    framePadding.y = framePaddingY;

    ImGui::PushStyleColor( ImGuiCol_FrameBg, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActive ).getUInt32() );
    ImGui::PushStyleColor( ImGuiCol_FrameBgHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered ).getUInt32() );
    ImGui::PushStyleColor( ImGuiCol_FrameBgActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered ).getUInt32() );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, framePadding );
    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::Small ) );
    if ( ImGui::BeginCombo( "##ActiveStatesComboBox", "", ImGuiComboFlags_NoArrowButton ) )
    {
        if ( activeBlockingItem_.item )
        {
            auto caption = activeBlockingItem_.item->name();
            closeBlocking = ImGui::SmallButton( ( "Close##" + caption ).c_str() );
            auto activeIt = RibbonSchemaHolder::schema().items.find( caption );
            if ( activeIt != RibbonSchemaHolder::schema().items.end() )
            {
                if ( !activeIt->second.caption.empty() )
                    caption = activeIt->second.caption;
            }
            ImGui::SameLine( 0, 5.0f );
            ImGui::Text( "%s", caption.c_str() );
        }
        for ( int i = 0; i < activeNonBlockingItems_.size(); ++i )
        {
            const auto& activeNonBlock = activeNonBlockingItems_[i];
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + 3.0f ); //spacing (ImGui::Spacing does not work here)
            auto caption = activeNonBlock.item->name();
            closeNonBlocking[i] = ImGui::SmallButton( ( "Close##" + caption ).c_str() );
            auto activeIt = RibbonSchemaHolder::schema().items.find( caption );
            if ( activeIt != RibbonSchemaHolder::schema().items.end() )
            {
                if ( !activeIt->second.caption.empty() )
                    caption = activeIt->second.caption;
            }
            ImGui::SameLine( 0, 5.0f );
            ImGui::Text( "%s", caption.c_str() );

        }
        ImGui::EndCombo();
    }
    if ( closeBlocking )
        itemPressed_( activeBlockingItem_.item, true );
    for ( int i = 0; i < activeNonBlockingItems_.size(); ++i )
        if ( closeNonBlocking[i] )
            itemPressed_( activeNonBlockingItems_[i].item, true );

    ImGui::PopFont();
    const char* text = "Active";
    ImGui::SetCursorPosX( xPos + ( btnSize - textSize ) * 0.5f );
    ImGui::SetCursorPosY( 2 * cTabYOffset * scaling );
    ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActiveText ).getUInt32() );
    ImGui::RenderText( ImGui::GetCursorPos(), text, text + 6, false );
    ImGui::PopStyleVar( 2 );
    ImGui::PopStyleColor( 4 );
}

bool RibbonMenu::drawGroupUngroupButton_( const std::vector<std::shared_ptr<Object>>& selected )
{
    bool someChanges = false;
    if ( selected.empty() )
        return someChanges;

    Object* parentObj = selected[0]->parent();
    bool canGroup = parentObj != nullptr;
    for ( int i = 1; i < selected.size(); ++i )
    {
        if ( selected[i]->parent() != parentObj )
        {
            canGroup = false;
            break;
        }
    }

    if ( canGroup && RibbonButtonDrawer::GradientButton( "Group", ImVec2( -1, 0 ) ) )
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
    if ( canUngroup && RibbonButtonDrawer::GradientButton( "Ungroup", ImVec2( -1, 0 ) ) )
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

bool RibbonMenu::onKeyUp_( int key, int modifiers )
{
    showShortcuts_ = false;
    return ImGuiMenu::onKeyUp_( key, modifiers );
}

void RibbonMenu::drawBigButtonItem_( const MenuItemInfo& item )
{
    auto width = buttonDrawer_.calcItemWidth( item, DrawButtonParams::SizeType::Big );

    auto availReg = ImGui::GetContentRegionAvail();

    const auto& style = ImGui::GetStyle();
    ImVec2 itemSize = ImVec2( width.baseWidth, availReg.y - 2 * style.WindowPadding.y );

    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + availReg.y * 0.5f - itemSize.y * 0.5f - ImGui::GetStyle().CellPadding.y * 0.5f );

    buttonDrawer_.drawButtonItem( item, { DrawButtonParams::SizeType::Big,itemSize,fontManager_.getFontSizeByType( RibbonFontManager::FontType::Icons ) } );
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
    const float screenWidth = float( getViewerInstance().window_width ) - ImGui::GetCursorScreenPos().x -
        float( groupsInTab.size() ) - 1.0f;
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
    if ( item->isActive() || available )
        item->action();
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
    std::string requirements;
    if ( !activeBlockingItem_.item || !item->blocking() )
    {
        requirements = item->isAvailable( selectedObjectsCache_ );
    }
    else
    {
        if ( activeBlockingItem_.item != item )
            requirements = "Other state is active";
    }
    return requirements;
}

void RibbonMenu::drawSceneListButtons_()
{
    auto menuScaling = menu_scaling();
    const float size = ( cMiddleIconSize + 8.f ) * menuScaling;
    const ImVec2 smallItemSize = { size, size };

    const DrawButtonParams params{ DrawButtonParams::SizeType::Small, smallItemSize, cMiddleIconSize };

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

std::vector<RibbonMenu::SearchResult> RibbonMenu::search_()
{
    std::vector<SearchResult> res;
    if ( searchLine_.empty() )
        return res;
    std::vector<std::pair<float, SearchResult>> resultListForSort;
    auto checkItem = [&] ( const MenuItemInfo& item, int t )
    {
        const auto& caption = item.caption.empty() ? item.item->name() : item.caption;
        const auto& tooltip = item.tooltip;
        float res = 0.0f;
        auto captionRes = findSubstringCaseInsensitive( caption, searchLine_ );
        auto tooltipRes = findSubstringCaseInsensitive( tooltip, searchLine_ );
        if ( captionRes == std::string::npos && tooltipRes == std::string::npos )
            return;
        if ( captionRes != std::string::npos )
            res += ( 1.0f - float( captionRes ) / float( caption.size() ) );
        if ( tooltipRes != std::string::npos )
            res += 0.5f * ( 1.0f - float( tooltipRes ) / float( tooltip.size() ) );
        resultListForSort.push_back( { res, SearchResult{t,&item} } );
    };
    const auto& schema = RibbonSchemaHolder::schema();
    auto lookUpMenuItemList = [&] ( const MenuItemsList& list, int t )
    {
        for ( int i = 0; i < list.size(); ++i )
        {
            auto item = schema.items.find( list[i] );
            if ( item == schema.items.end() )
                continue;
            if ( !item->second.item )
                continue;
            checkItem( item->second, t );
            if ( item->second.item->type() == RibbonItemType::ButtonWithDrop )
            {
                for ( const auto& dropRibItem : item->second.item->dropItems() )
                {
                    if ( !dropRibItem )
                        continue;
                    if ( std::dynamic_pointer_cast< LambdaRibbonItem >( dropRibItem ) )
                        continue;
                    auto dropItem = schema.items.find( dropRibItem->name() );
                    if ( dropItem == schema.items.end() )
                        continue;
                    checkItem( dropItem->second, t );
                }
            }
        }
    };
    for ( int t = 0; t < schema.tabsOrder.size(); ++t )
    {
        auto tabItem = schema.tabsMap.find( schema.tabsOrder[t] );
        if ( tabItem == schema.tabsMap.end() )
            continue;
        for ( int g = 0; g < tabItem->second.size(); ++g )
        {
            auto groupItem = schema.groupsMap.find( schema.tabsOrder[t] + tabItem->second[g] );
            if ( groupItem == schema.groupsMap.end() )
                continue;
            lookUpMenuItemList( groupItem->second, t );
        }
    }
    lookUpMenuItemList( schema.headerQuickAccessList, -1 );
    lookUpMenuItemList( schema.sceneButtonsList, -1 );

    std::sort( resultListForSort.begin(), resultListForSort.end(), [] ( const auto& a, const auto& b )
    {
        return intptr_t( a.second.item ) < intptr_t( b.second.item );
    } );
    resultListForSort.erase( 
        std::unique( resultListForSort.begin(), resultListForSort.end(), 
                     [] ( const auto& a, const auto& b )
    {
        return a.second.item == b.second.item;
    } ),
        resultListForSort.end() );

    std::sort( resultListForSort.begin(), resultListForSort.end(), [] ( const auto& a, const auto& b )
    {
        return a.first > b.first;
    } );
    res.reserve( resultListForSort.size() );
    for ( const auto& sortedRes : resultListForSort )
        res.push_back( sortedRes.second );

    return res;
}

void RibbonMenu::readMenuItemsStructure_()
{
    RibbonSchemaLoader loader;
    loader.loadSchema();
    quickAccessList_ = RibbonSchemaHolder::schema().defaultQuickAccessList;
}

void RibbonMenu::postResize_( int width, int height )
{
    Menu::postResize_( width, height );
    fixViewportsSize_( width, height );
}

void RibbonMenu::postRescale_( float x, float y )
{
    Menu::postRescale_( x, y );
    buttonDrawer_.setScaling( menu_scaling() );
    fixViewportsSize_( Viewer::instanceRef().window_width, Viewer::instanceRef().window_height );

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
            statePlugin->drawDialog( menu_scaling(), ImGui::GetCurrentContext() );

            if ( !itemPtr.dialogPositionFixed )
            {
                itemPtr.dialogPositionFixed = true;
                auto* window = ImGui::FindWindowByName( itemPtr.item->name().c_str() ); // this function is hidden in imgui_internal.h
                // viewer->window_width here because ImGui use screen space
                if ( window )
                {
                    ImVec2 pos = ImVec2( viewer->window_width - window->Size.x, float( topPanelOpenedHeight_ ) * menu_scaling() );
                    ImGui::SetWindowPos( window, pos, ImGuiCond_Always );
                }
            }

            if ( !statePlugin->dialogIsOpen() )
            {
                itemPtr.item->action();
            }
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
    sceneSize_.x = std::min( sceneSize_.x, viewerRef.window_width - 100 * scaling );
    sceneSize_.y = viewerRef.window_height + 2.0f - float( currentTopPanelHeight_ ) * scaling;
    ImGui::SetWindowSize( "RibbonScene", sceneSize_, ImGuiCond_Always );
    ImGui::SetNextWindowSizeConstraints( ImVec2( 100 * scaling, -1.f ), ImVec2( viewerRef.window_width / 2.f, -1.f ) ); // TODO take out limits to special place
    ImGui::PushStyleVar( ImGuiStyleVar_Alpha, 1.f );
    auto colorBg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    colorBg.w = 1.f;
    ImGui::PushStyleColor( ImGuiCol_WindowBg, colorBg );

    ImGui::Begin(
        "RibbonScene", nullptr,
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
    );
    drawRibbonSceneListContent_( selectedObjs, allObj );
    drawRibbonSceneInformation_( selectedObjs );

    const auto newSize = ImGui::GetWindowSize();
    if ( fixFirstGetSize_ && ( newSize.x != sceneSize_.x || newSize.y != sceneSize_.y ) )
    {
        sceneSize_ = newSize;
        fixViewportsSize_( viewerRef.window_width, viewerRef.window_height );
    }
    fixFirstGetSize_ = true;

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void RibbonMenu::drawRibbonSceneListContent_( std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all )
{
    drawSceneListButtons_();
    // mesh with index 0 is Ancillary, and cannot be removed
    // it can be cleaned but it is inconsistent, so this mesh is untouchable
    int uniqueCounter = 0;
    ImGui::BeginChild( "Meshes", ImVec2( -1, -( informationHeight_ + transformHeight_ ) ), false );
    auto children = SceneRoot::get().children();
    for ( const auto& child : children )
        draw_object_recurse_( *child, selected, all, uniqueCounter );
    makeDragDropTarget_( SceneRoot::get(), false, true, uniqueCounter + 1 );
    //ImGui::SetWindowSize( ImVec2(size.x,0) );

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

void RibbonMenu::drawRibbonViewportsLabels_()
{
    const auto scaling = menu_scaling();
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::SemiBold ) );
    for ( const auto& vp : viewer->viewport_list )
    {
        std::string windowName = "##ProjectionMode" + std::to_string( vp.id.value() );
        auto pos = viewer->viewportToScreen( Vector3f( 0.0f, height( vp.getViewportRect() ) - 50.0f * scaling, 0.0f ), vp.id );
        ImGui::SetNextWindowPos( ImVec2( pos.x, pos.y ) );
        constexpr std::array<const char*, 2> cProjModeString = { "Orthographic" , "Perspective" };
        ImGui::Begin( windowName.c_str(), nullptr,
                      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize |
                      ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus );
        if ( viewer->viewport_list.size() > 1 )
            ImGui::Text( "Viewport Id: %d, %s", vp.id.value(), cProjModeString[int( !vp.getParameters().orthographic )] );
        else
            ImGui::Text( "%s", cProjModeString[int( !vp.getParameters().orthographic )] );
        ImGui::End();
    }
    ImGui::PopFont();
    ImGui::PopStyleVar();
}

void RibbonMenu::drawRibbonSceneInformation_( std::vector<std::shared_ptr<Object>>& /*selected*/ )
{
    informationHeight_ = drawSelectionInformation_();

    transformHeight_ = drawTransform_();
}

void RibbonMenu::drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& selected )
{
    const auto selectedVisualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( ImGui::BeginPopupContextItem() )
    {
        bool wasChanged = false;
        if ( selectedVisualObjs.empty() )
        {
            wasChanged |= drawGeneralOptions_( selected );
            wasChanged |= drawRemoveButton_( selected );
            wasChanged |= drawGroupUngroupButton_( selected );
        }
        else if ( ImGui::BeginTable( "##DrawOptions", 2, ImGuiTableFlags_BordersInnerV ) )
        {
            ImGui::TableNextColumn();
            wasChanged |= drawGeneralOptions_( selected );
            wasChanged |= drawDrawOptionsCheckboxes_( selectedVisualObjs );
            ImGui::TableNextColumn();
            wasChanged |= drawDrawOptionsColors_( selectedVisualObjs );
            wasChanged |= drawRemoveButton_( selected );
            wasChanged |= drawGroupUngroupButton_( selected );
            ImGui::EndTable();
        }
        if ( wasChanged )
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

const char* RibbonMenu::getSceneItemIconByTypeName_( const std::string& typeName ) const
{
    if ( typeName == ObjectMesh::TypeName() )
        return "\xef\x82\xac";
#ifndef __EMSCRIPTEN__
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

void RibbonMenu::drawQuickAccessMenu_()
{
    auto menuScaling = menu_scaling();
    auto windowPadding = ImVec2( 12 * menuScaling, 4 * menuScaling );
    auto itemSpacing = ImVec2( 12 * menuScaling, 0 );
    const ImVec2 smallItemSize = { cQuickAccessBarHeight * menuScaling - 2.0f * windowPadding.y, cQuickAccessBarHeight * menuScaling - 2.0f * windowPadding.y };

    int itemCount = 0;
    int droppedItemCount = 0;
    //TODO calc if list changes
    for ( const auto& item : quickAccessList_ )
    {
        auto it = RibbonSchemaHolder::schema().items.find( item );
        if ( it == RibbonSchemaHolder::schema().items.end() )
            continue;
        ++itemCount;
        if ( it->second.item->type() == RibbonItemType::ButtonWithDrop )
            ++droppedItemCount;
    }

    if ( !itemCount ) return;

    const float windowWidth = windowPadding.x * 2
        + smallItemSize.x * itemCount
        + smallItemSize.x * cSmallItemDropSizeModifier * droppedItemCount
        + itemSpacing.x * ( itemCount - 1 );

    if ( windowWidth >= getViewerInstance().window_width - sceneSize_.x )
        return; // dont show quick panel if window is too small

    const float windowPosX = std::max( getViewerInstance().window_width / 2.f - windowWidth / 2.f, sceneSize_.x - 1.0f );

    ImGui::SetNextWindowPos( ImVec2( windowPosX, float( currentTopPanelHeight_ ) * menuScaling - 1 ) );
    ImGui::SetNextWindowSize( ImVec2( windowWidth, cQuickAccessBarHeight * menuScaling ), ImGuiCond_Always );

    ImGui::PushStyleColor( ImGuiCol_WindowBg, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::QuickAccessBackground ).getUInt32() );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, windowPadding );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 1.0f );
    ImGui::Begin(
        "QuickAccess", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoFocusOnAppearing
    );
    ImGui::PopStyleVar( 2 );
    ImGui::PopStyleColor();

    DrawButtonParams params{ DrawButtonParams::SizeType::Small, smallItemSize, cMiddleIconSize,DrawButtonParams::RootType::Toolbar };

    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::Small ) );
    for ( const auto& item : quickAccessList_ )
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

    ImGui::PopStyleVar();
    ImGui::PopFont();
    ImGui::End();
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

void RibbonMenu::setupShortcuts_()
{
    if ( !shortcutManager_ )
        shortcutManager_ = std::make_shared<ShortcutManager>();

    shortcutManager_->setShortcut( { GLFW_KEY_H,0 }, { "Toggle selected objects visibility",[] ()
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
    shortcutManager_->setShortcut( { GLFW_KEY_F1,0 }, { "Show this help",[this] ()
   {
       showShortcuts_ = true;
   } } );
    shortcutManager_->setShortcut( { GLFW_KEY_D,0 }, { "Toggle statistics window",[this] ()
    {
        showStatistics_ = !showStatistics_;
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_F,0 }, { "Toggle shading of selected objects",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( MeshVisualizePropertyType::FlatShading, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_I,0 }, { "Invert normals of selected objects",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( VisualizeMaskType::InvertedNormals, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_L,0 }, { "Toggle edges on selected meshes",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
                sel->toggleVisualizeProperty( MeshVisualizePropertyType::Edges, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_O,0 }, { "Toggle orthographic in current viewport",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        viewport.setOrthographic( !viewport.getParameters().orthographic );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_T,0 }, { "Toggle faces on selected meshes",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto selected = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( MeshVisualizePropertyType::Faces, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_DOWN,0 }, { "Select next object",[] ()
    {
        changeSelection( true,0 );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_DOWN,GLFW_MOD_SHIFT }, { "Add next object to selection",[] ()
    {
        changeSelection( true,GLFW_MOD_SHIFT );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_UP,0 }, { "Select previous object",[] ()
    {
        changeSelection( false,0 );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_UP,GLFW_MOD_SHIFT }, { "Add previous object to selection",[] ()
    {
        changeSelection( false,GLFW_MOD_SHIFT );
    } } );

    auto addShortcut = [this] ( std::string pluginName, const ShortcutManager::ShortcutKey& key )
    {
        auto pluginIt = RibbonSchemaHolder::schema().items.find( pluginName );
        if ( pluginIt != RibbonSchemaHolder::schema().items.end() )
        {
            auto caption = pluginIt->second.caption.empty() ? pluginIt->first : pluginIt->second.caption;
            shortcutManager_->setShortcut( key, { caption,[item = pluginIt->second.item, this]()
            {
                itemPressed_( item, getRequirements_( item ).empty() );
            } } );
        }
#ifndef __EMSCRIPTEN__
        else
            assert( !"item not found" );
#endif
    };

    addShortcut( "Ribbon Scene Select all", { GLFW_KEY_A, GLFW_MOD_CONTROL } );
    addShortcut( "Undo", { GLFW_KEY_Z, GLFW_MOD_CONTROL } );
    addShortcut( "Redo", { GLFW_KEY_Z, GLFW_MOD_CONTROL | GLFW_MOD_SHIFT } );
    addShortcut( "Fit data", { GLFW_KEY_F, GLFW_MOD_CONTROL } );
    addShortcut( "Select objects", { GLFW_KEY_Q, GLFW_MOD_CONTROL } );
    addShortcut( "Face Selector", { GLFW_KEY_W, GLFW_MOD_CONTROL } );
    addShortcut( "Clear Selections", { GLFW_KEY_T, GLFW_MOD_CONTROL } );
    addShortcut( "Open files", { GLFW_KEY_O, GLFW_MOD_CONTROL } );
    addShortcut( "Save Scene", { GLFW_KEY_S, GLFW_MOD_CONTROL } );
    addShortcut( "Save Scene As", { GLFW_KEY_S, GLFW_MOD_CONTROL | GLFW_MOD_SHIFT } );
    addShortcut( "New", { GLFW_KEY_N, GLFW_MOD_CONTROL } );
    addShortcut( "Delete Selection", { GLFW_KEY_DELETE, 0 } );
    addShortcut( "Ribbon Scene Show only previous", { GLFW_KEY_F3, 0 } );
    addShortcut( "Ribbon Scene Show only next", { GLFW_KEY_F4, 0 } );
    addShortcut( "New", { GLFW_KEY_N, GLFW_MOD_CONTROL } );
    addShortcut( "Ribbon Scene Rename", { GLFW_KEY_F2, 0 } );
    addShortcut( "Ribbon Scene Remove selected objects", { GLFW_KEY_R, GLFW_MOD_SHIFT } );
}

void RibbonMenu::beginTopPanel_()
{
    const auto scaling = menu_scaling();
    ImGui::SetNextWindowPos( ImVec2( 0, 0 ) );
    ImGui::SetNextWindowSize( ImVec2( ( float ) Viewer::instanceRef().window_width, currentTopPanelHeight_ * scaling ) );

    ImGui::PushStyleVar( ImGuiStyleVar_Alpha, 1.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, 5.0f * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_ChildRounding, 5.0f * scaling );
    auto colorBg = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TopPanelBackground );
    if ( collapseState_ != CollapseState::Opened )
        colorBg.a = 255;
    else
        colorBg.a = 228;
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

    drawHeaderPannel_();

    // tab content position
    ImGui::SetCursorPosX( style.CellPadding.x * 2 );
    ImGui::SetCursorPosY( ( cTabYOffset + cTabHeight ) * menu_scaling() + 2 );

    ImGuiTableFlags tableFlags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerV;

    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::Small ) );
    const auto& tab = RibbonSchemaHolder::schema().tabsOrder[activeTabIndex_];
    if ( collapseState_ != CollapseState::Closed )
    {
        auto tabIt = RibbonSchemaHolder::schema().tabsMap.find( tab );
        if ( tabIt != RibbonSchemaHolder::schema().tabsMap.end() )
        {
            ImGui::PushStyleColor( ImGuiCol_TableBorderLight, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Borders ).getUInt32() );
            ImGui::PushStyleColor( ImGuiCol_ScrollbarBg, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TopPanelBackground ).getUInt32() );
            ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, cellPadding );
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );
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
            ImGui::PopStyleVar( 3 );
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

        rect.min.x = ( rect.min.x - viewportsBounds.min.x ) / minMaxDiff.x * ( width - sceneWidth ) + sceneWidth;
        rect.min.y = ( rect.min.y - viewportsBounds.min.y ) / minMaxDiff.y * ( height - ( topPanelHeightScaled - 1 ) ); // -1 - buffer pixel
        rect.max.x = rect.min.x + MR::width( rect ) / minMaxDiff.x * ( width - sceneWidth );
        rect.max.y = rect.min.y + MR::height( rect ) / minMaxDiff.y * ( height - ( topPanelHeightScaled - 1 ) ); // -1 - buffer pixel
        if ( MR::width( rect ) <= 0 || MR::height( rect ) <= 0 )
            continue;
        vp.setViewportRect( rect );
    }
}

}
