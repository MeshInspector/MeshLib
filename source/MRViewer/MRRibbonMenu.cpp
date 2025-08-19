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
#include "MRUITestEngine.h"
#include "MRViewerSettingsManager.h"
#include "MRUIStyle.h"
#include "MRViewport.h"
#include "MRViewer.h"
#include "MRSceneCache.h"
#include "MRShortcutManager.h"
#include "MRMouseController.h"
#include "MRRibbonSceneObjectsListDrawer.h"
#include "MRClipboard.h"
#include "MRSceneOperations.h"
#include "MRToolbar.h"
#include "MRMesh/MRObjectsAccess.h"
#include <MRMesh/MRString.h>
#include <MRMesh/MRSystem.h>
#include <MRMesh/MRStringConvert.h>
#include <MRMesh/MRSerializer.h>
#include <MRMesh/MRObjectsAccess.h>
#include <MRMesh/MRChangeXfAction.h>
#include <MRSymbolMesh/MRObjectLabel.h>
#include <MRMesh/MRChangeSceneObjectsOrder.h>
#include <MRMesh/MRChangeSceneAction.h>
#include <MRMesh/MRChangeObjectFields.h>
#include <MRMesh/MRSceneRoot.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRObjectLines.h>
#include <MRMesh/MRObjectDistanceMap.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTimer.h>
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

// Modifier for shortcuts
// Some shortcuts still use GLFW_MOD_CONTROL on Mac to avoid conflict with system shortcuts
#if !defined( __APPLE__ )
#define CONTROL_OR_SUPER GLFW_MOD_CONTROL
#else
#define CONTROL_OR_SUPER GLFW_MOD_SUPER
#endif

namespace MR
{

namespace
{

constexpr auto cTransformContextName = "TransformContextWindow";

auto getItemCaption( const std::string& name )->const std::string&
{
    auto it = RibbonSchemaHolder::schema().items.find( name );
    if ( it == RibbonSchemaHolder::schema().items.end() )
        return name;
    return  it->second.caption.empty() ? name : it->second.caption;
}

} //anonymous namespace

RibbonMenu::RibbonMenu() :
    toolbar_( new Toolbar() )
{
}

RibbonMenu::~RibbonMenu()
{
}

std::shared_ptr<RibbonMenu> RibbonMenu::instance()
{
    return getViewerInstance().getRibbonMenu();
}

void RibbonMenu::setCustomContextCheckbox(
    const std::string& name,
    CustomContextMenuCheckbox customContextMenuCheckbox )
{
    customCheckBox_[name] = customContextMenuCheckbox;
}

void RibbonMenu::init( MR::Viewer* _viewer )
{
    MR_TIMER;
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
        const bool cShowTopPanel = menuUIConfig_.topLayout != RibbonTopPanelLayoutMode::None;
        const bool cShowAny = cShowTopPanel || menuUIConfig_.drawScenePanel;

        if ( cShowTopPanel )
        {
            drawTopPanel_( menuUIConfig_.topLayout == RibbonTopPanelLayoutMode::RibbonWithTabs, menuUIConfig_.centerRibbonItems );

            drawActiveBlockingDialog_();
            drawActiveNonBlockingDialogs_();
        }

        if ( cShowTopPanel && menuUIConfig_.drawToolbar )
        {
            toolbar_->drawToolbar();
            toolbar_->drawCustomize();
        }

        if ( menuUIConfig_.drawScenePanel )
            drawRibbonSceneList_();

        if ( menuUIConfig_.drawViewportTags )
            drawRibbonViewportsLabels_();

        if ( cShowTopPanel )
            drawActiveList_();

        if ( cShowAny )
            draw_helpers();

        if ( menuUIConfig_.drawNotifications )
            drawNotifications_();

        prevFrameSelectedObjectsCache_ = SceneCache::getAllObjects<const Object, ObjectSelectivityType::Selected>();
    };

    buttonDrawer_.setMenu( this );
    buttonDrawer_.setShortcutManager( getShortcutManager().get() );
    buttonDrawer_.setScaling( menu_scaling() );
    buttonDrawer_.setOnPressAction( [&] ( std::shared_ptr<RibbonMenuItem> item, const std::string& req )
    {
        itemPressed_( item, req );
    } );
    buttonDrawer_.setGetterRequirements( [&] ( std::shared_ptr<RibbonMenuItem> item )
    {
        return getRequirements_( item );
    } );

    toolbar_->setRibbonMenu( this );
    std::shared_ptr<RibbonSceneObjectsListDrawer> ribbonObjectsSceneListDrawer = std::make_shared<RibbonSceneObjectsListDrawer>();
    ribbonObjectsSceneListDrawer->initRibbonMenu( this );
    sceneObjectsList_ = std::dynamic_pointer_cast< SceneObjectsListDrawer >( ribbonObjectsSceneListDrawer );
    searcher_.setRequirementsFunc( [this] ( const std::shared_ptr<RibbonMenuItem>& item )->std::string
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
    fontManager_.initFontManagerInstance( nullptr );
    ImGuiMenu::shutdown();
    RibbonIcons::free();
}

void RibbonMenu::openToolbarCustomize()
{
    toolbar_->openCustomize();
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

void RibbonMenu::setQuickAccessListVersion( int version )
{
    toolbar_->setItemsListVersion( version );
}

void RibbonMenu::readQuickAccessList( const Json::Value& root )
{
    toolbar_->readItemsList( root );
}

void RibbonMenu::resetQuickAccessList()
{
    toolbar_->resetItemsList();
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
                itemPressed_( activeBlockingItem_.item );
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
            for ( auto& it : activeNonBlockingItems_ )
                if ( it.item == item )
                    it.item = {}; // do not erase while we could be iterating over this item right now, it will be removed from list after it
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
    searcher_.drawMenuUI( { buttonDrawer_, fontManager_, [this] ( int i ) { changeTab_( i ); }, menu_scaling() } );
}

void RibbonMenu::drawCollapseButton_()
{
    const auto scaling = menu_scaling();
    auto font = fontManager_.getFontByType( RibbonFontManager::FontType::Icons );
    font->Scale = 0.7f;

    float btnSize = scaling * cTopPanelAditionalButtonSize;

    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );

    if ( collapseState_ == CollapseState::Pinned )
    {
        ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText ).getUInt32() );
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
        ImGui::PopStyleColor();
        UI::setTooltipIfHovered( "Unpin", scaling );
    }
    else
    {
        ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText ).getUInt32() );
        ImGui::PushFont( font );
        if ( ImGui::Button( "\xef\x81\xb7", ImVec2( btnSize, btnSize ) ) )
        {
            collapseState_ = CollapseState::Pinned;
            fixViewportsSize_( getViewerInstance().framebufferSize.x, getViewerInstance().framebufferSize.y );
        }
        ImGui::PopFont();
        ImGui::PopStyleColor();
        UI::setTooltipIfHovered( "Pin", scaling );
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

void RibbonMenu::drawHelpButton_( const std::string& url )
{
    const auto scaling = menu_scaling();
    auto font = fontManager_.getFontByType( RibbonFontManager::FontType::Icons );
    font->Scale = 0.7f;

    float btnSize = scaling * cTopPanelAditionalButtonSize;

    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding * scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );

    ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText ).getUInt32() );
    ImGui::PushFont( font );
    if ( ImGui::Button( "\xef\x81\x99", ImVec2( btnSize, btnSize ) ) )
        OpenLink( url );
    ImGui::PopFont();
    ImGui::PopStyleColor();
    UI::setTooltipIfHovered( "Open help page", scaling );
    font->Scale = 1.0f;

    ImGui::PopStyleColor( 3 );
    ImGui::PopStyleVar( 2 );
}

bool RibbonMenu::drawCustomCheckBox( const std::vector<std::shared_ptr<Object>>& selected, SelectedTypesMask selectedMask )
{
    bool res = false;
    for ( auto& [name, custom] : customCheckBox_ )
    {
        if ( !bool( selectedMask ) || bool( ~custom.selectedMask & selectedMask ) )
        {
            continue;
        }

        bool atLeastOneTrue = false;
        bool allTrue = true;
        for ( auto& obj : selected )
        {
            if ( !obj )
            {
                continue;
            }

            bool isThisTrue = custom.getter( obj, viewer->viewport().id );
            atLeastOneTrue = atLeastOneTrue || isThisTrue;
            allTrue = allTrue && isThisTrue;
        }

        std::pair<bool, bool> realRes{ atLeastOneTrue, allTrue };

        if ( UI::checkboxMixed( name.c_str(), &realRes.first, !realRes.second && realRes.first ) )
        {
            for ( auto& obj : selected )
            {
                if ( !obj )
                {
                    continue;
                }

                custom.setter( obj, viewer->viewport().id, realRes.first );
            }
            res = true;
        }
    }

    return res;
}

void RibbonMenu::sortObjectsRecursive_( std::shared_ptr<Object> object )
{
    auto& children = object->children();
    for ( const auto& child : children )
        sortObjectsRecursive_( child );

    AppendHistory( std::make_shared<ChangeSceneObjectsOrder>( "Sort object children", object ) );
    object->sortChildren();
}

void RibbonMenu::drawHeaderQuickAccess_( float menuScaling )
{
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
    
    auto cursorPos = ImGui::GetCursorPos();
    ImGui::SetCursorPosX( cursorPos.x + itemSpacing.x );
    ImGui::SetCursorPosY( cursorPos.y + itemSpacing.y );

    DrawButtonParams params{ DrawButtonParams::SizeType::Small, ImVec2( itemSize,itemSize ), iconSize,DrawButtonParams::RootType::Header };

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding * menuScaling );
    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::Small ) );
    UI::TestEngine::pushTree( "QuickAccess" );
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
    UI::TestEngine::popTree(); // "QuickAccess"
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

    drawHeaderQuickAccess_( menuScaling );

    ImGui::PushFont( fontManager_.getFontByType( RibbonFontManager::FontType::SemiBold ) );
    // TODO_store: this needs recalc only on scaling change, no need to calc each frame
    const auto& schema = RibbonSchemaHolder::schema();
    std::vector<float> textSizes( schema.tabsOrder.size() );// TODO_store: add to some store at the beginning not to calc each time
    std::vector<float> tabSizes( schema.tabsOrder.size() );// TODO_store: add to some store at the beginning not to calc each time
    auto summaryTabPannelSize = 2 * 12.0f * menuScaling - cTabsInterval * menuScaling; // init shift (by eye, not defined in current design maket)
    for ( int i = 0; i < tabSizes.size(); ++i )
    {
        if ( schema.tabsOrder[i].experimental && !getViewerInstance().experimentalFeatures )
            continue;
        const auto& tabStr = schema.tabsOrder[i].name;
        textSizes[i] = ImGui::CalcTextSize( tabStr.c_str() ).x;
        tabSizes[i] = std::max( textSizes[i] + cTabLabelMinPadding * 2 * menuScaling, cTabMinimumWidth * menuScaling );
        summaryTabPannelSize += ( tabSizes[i] + cTabsInterval * menuScaling );
    }

    float availWidth = 0.0f;
    {
        auto backupPos = ImGui::GetCursorPos();
        ImGui::PopStyleVar( 2 ); // draw helpers with default style
        availWidth = drawHeaderHelpers_( summaryTabPannelSize, menuScaling );
        // push header panel style back
        ImGui::PushStyleVar( ImGuiStyleVar_TabRounding, cTabFrameRounding * menuScaling );
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 0, 0 ) );
        ImGui::SetCursorPos( backupPos );
    }

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
        if ( buttonDrawer_.drawTabArrowButton( "\xef\x81\x88", ImVec2( cTopPanelScrollBtnSize * menuScaling, ( cTabYOffset + cTabHeight ) * menuScaling ), btnSize ) )
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
    UI::TestEngine::pushTree( "RibbonTabs" );
    ImGui::PopStyleVar();
    auto window = ImGui::GetCurrentContext()->CurrentWindow;

    auto basePos = window->Pos;
    if ( needScroll )
    {
        basePos.x -= tabPanelScroll_;
    }
    basePos.x += 12.0f * menuScaling;// temp hardcoded offset
    basePos.y = cTabYOffset * menuScaling - 1;// -1 due to ImGui::TabItemBackground internal offset
    for ( int i = 0; i < schema.tabsOrder.size(); ++i )
    {
        if ( schema.tabsOrder[i].experimental && !getViewerInstance().experimentalFeatures )
        {
            if ( activeTabIndex_ == i )
            {
                activeTabIndex_ = ( activeTabIndex_ + 1 ) % int( schema.tabsOrder.size() );
            }
            continue;
        }
        const auto& tabStr = schema.tabsOrder[i].name;
        const auto& tabWidth = tabSizes[i];
        ImVec2 tabBbMaxPoint( basePos.x + tabWidth, basePos.y + cTabHeight * menuScaling + 2 ); // +2 due to TabItemBackground internal offset
        ImRect tabRect( basePos, tabBbMaxPoint );
        std::string strId = "##" + tabStr + "TabId"; // TODO_store: add to some store at the beginning not to calc each time
        auto tabId = window->GetID( strId.c_str() );
        ImGui::ItemAdd( tabRect, tabId );
        bool hovered, held;
        bool pressed = ImGui::ButtonBehavior( tabRect, tabId, &hovered, &held );
        pressed = UI::TestEngine::createButton( tabStr ) || pressed; // Must not short-circuit.
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
    UI::TestEngine::popTree(); // "RibbonTabs"
    if ( needFwdBtn )
    {
        ImGui::SameLine();
        ImGui::SetCursorPosX( ImGui::GetCursorPosX() + cTabsInterval * menuScaling );
        const float btnSize = 0.5f * fontManager_.getFontSizeByType( RibbonFontManager::FontType::Icons );
        if ( buttonDrawer_.drawTabArrowButton( "\xef\x81\x91", ImVec2( cTopPanelScrollBtnSize * menuScaling, ( cTabYOffset + cTabHeight ) * menuScaling ), btnSize ) )
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

}

float RibbonMenu::drawHeaderHelpers_( float requiredTabSize, float menuScaling )
{
    // prepare active button
    bool needActive = hasAnyActiveItem() && toolbar_->getCurrentToolbarWidth() == 0.0f;
    float activeBtnSize = cTabHeight * menuScaling - 4 * menuScaling; // small offset from border

    // 40 - active button size (optional)
    // 40 - help button size
    // 40 - search button size
    // 40 - collapse button size
    auto availWidth = ImGui::GetContentRegionAvail().x - ( ( needActive ? 3 : 2 ) * 40.0f ) * menuScaling;
    searcher_.setSmallUI( availWidth - requiredTabSize < searcher_.getSearchStringWidth() * menuScaling );
    auto searcherWidth = searcher_.getWidthMenuUI();
    availWidth -= searcherWidth * menuScaling;

    if ( needActive )
    {
        ImGui::SetCursorPos( ImVec2( float( getViewerInstance().framebufferSize.x ) -
            ( 110 + searcherWidth ) * menuScaling, cTabYOffset * menuScaling ) );
        drawActiveListButton_( activeBtnSize );
    }

    ImGui::SetCursorPos( ImVec2( float( getViewerInstance().framebufferSize.x ) - ( 70.f + searcherWidth ) * menuScaling, cTabYOffset * menuScaling ) );
    drawSearchButton_();

    ImGui::SetCursorPos( ImVec2( float( getViewerInstance().framebufferSize.x ) - 70.0f * menuScaling, cTabYOffset * menuScaling ) );
    drawHelpButton_( "https://meshinspector.com/inapphelp/" );

    ImGui::SetCursorPos( ImVec2( float( getViewerInstance().framebufferSize.x ) - 30.0f * menuScaling, cTabYOffset * menuScaling ) );
    drawCollapseButton_();

    return availWidth;
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
            auto btnText = "Close" + childName;
            if ( UI::button( btnText.c_str(), btnSize ) )
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
            itemPressed_( activeBlockingItem_.item );
        for ( int i = 0; i < activeNonBlockingItems_.size(); ++i )
            if ( closeNonBlocking[i] )
                itemPressed_( activeNonBlockingItems_[i].item );
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void RibbonMenu::drawNotifications_()
{
    auto scaling = menu_scaling();
    Box2i limitRect( Vector2i(), getViewerInstance().framebufferSize );
    limitRect.min.x = int( sceneSize_.x );
    limitRect.max.y -= int( currentTopPanelHeight_ * scaling );
    limitRect.min.y = int( ( StyleConsts::Notification::cWindowsPosY - StyleConsts::Notification::cWindowPadding - StyleConsts::Notification::cHistoryButtonSizeY ) * scaling );
    notifier_.draw( scaling, limitRect );
}

void RibbonMenu::setMenuUIConfig( const RibbonMenuUIConfig& newConfig )
{
    if ( menuUIConfig_ == newConfig )
        return;
    menuUIConfig_ = newConfig;
    fixViewportsSize_( getViewerInstance().framebufferSize.x, getViewerInstance().framebufferSize.y );
}

bool RibbonMenu::drawGroupUngroupButton( const std::vector<std::shared_ptr<Object>>& selected )
{
    bool someChanges = false;
    if ( selected.empty() )
        return someChanges;

    Object* parentObj = selected[0]->parent();
    bool canGroup = parentObj && selected.size() >= 2;
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

        SCOPED_HISTORY( "Group" );
        AppendHistory<ChangeSceneAction>( "Add object", group, ChangeSceneAction::Type::AddObject );
        parentObj->addChild( group );
        for ( int i = 0; i < selected.size(); ++i )
        {
            // for now do it by one object
            AppendHistory<ChangeSceneAction>( "Remove object", selected[i], ChangeSceneAction::Type::RemoveObject );
            selected[i]->detachFromParent();
            AppendHistory<ChangeSceneAction>( "Add object", selected[i], ChangeSceneAction::Type::AddObject );
            group->addChild( selected[i] );
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
    canUngroup = parentObj && std::all_of( selected.begin(), selected.end(),
        []( const std::shared_ptr<Object>& selObj ) { return !selObj->children().empty(); } );
    if ( canUngroup && UI::button( "Ungroup", Vector2f( -1, 0 ) ) )
    {
        someChanges |= true;
        SCOPED_HISTORY( "Ungroup" );
        for ( const auto& selObj : selected )
        {
            bool reorderDone = moveAllChildrenWithUndo( *selObj, *parentObj );
            assert( reorderDone );
            if ( reorderDone )
            {
                // remove group folder (now empty)
                auto ptr = std::dynamic_pointer_cast< VisualObject >( selObj );
                if ( !ptr && selObj->children().empty() )
                {
                    AppendHistory<ChangeSceneAction>( "Remove object", selObj, ChangeSceneAction::Type::RemoveObject );
                    selObj->detachFromParent();
                }
            }
        }
    }

    return someChanges;
}

void RibbonMenu::pushNotification( const RibbonNotification& notification )
{
    notifier_.pushNotification( notification );
}

void RibbonMenu::cloneTree( const std::vector<std::shared_ptr<Object>>& selectedObjects )
{
    const std::regex pattern( R"(.* Clone(?:| \([0-9]+\))$)" );
    SCOPED_HISTORY( "Clone" );
    for ( const auto& obj : selectedObjects )
    {
        if ( !obj )
            continue;
        auto cloneObj = obj->cloneTree();
        AppendHistory<ChangeObjectSelectedAction>( "unselect original", obj, false );
        AppendHistory<ChangeObjectVisibilityAction>( "hide original", obj, ViewportMask() );
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
    SCOPED_HISTORY( "Clone Selection" );
    std::shared_ptr<VisualObject> newObj;
    std::string name;
    if ( auto selectedMesh = std::dynamic_pointer_cast< ObjectMesh >( object ) )
    {
        if ( !selectedMesh->mesh() )
            return;
        newObj = cloneRegion( selectedMesh, selectedMesh->getSelectedFaces() );
        name = "ObjectMesh";
    }
    else if ( auto selectedPoints = std::dynamic_pointer_cast< ObjectPoints >( object ) )
    {
        if ( !selectedPoints->pointCloud() )
            return;
        newObj = cloneRegion( selectedPoints, selectedPoints->getSelectedPoints() );
        name = "ObjectPoints";
    }

    AppendHistory<ChangeObjectSelectedAction>( "unselect original", object, false );
    AppendHistory<ChangeObjectVisibilityAction>( "hide original", object, ViewportMask() );

    newObj->setName( object->name() + " Partial" );
    newObj->setXf( object->xf() );
    newObj->select( true );
    AppendHistory<ChangeSceneAction>( "Selection to New object: add " + name, newObj, ChangeSceneAction::Type::AddObject );
    object->parent()->addChild( newObj );
}

bool RibbonMenu::drawCloneButton( const std::vector<std::shared_ptr<Object>>& selected )
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

bool RibbonMenu::drawSelectSubtreeButton( const std::vector<std::shared_ptr<Object>>& selected )
{
    bool someChanges = false;
    const bool subtreeExists = std::any_of( selected.begin(), selected.end(), [] ( std::shared_ptr<Object> obj )
    {
        return obj && objectHasSelectableChildren( *obj );
    } );

    if ( selected.empty() || !subtreeExists )
        return someChanges;

    if ( UI::button( "Select Subtree", { -1, 0 } ) )
    {
        for ( auto selectedObject : selected )
        {
            std::stack<std::shared_ptr<Object>> objects;
            objects.push( selectedObject );
            while ( !objects.empty() )
            {
                auto object = objects.top();
                objects.pop();

                if ( !object )
                    continue;

                object->select( true );
                if ( sceneObjectsList_->getShowNewSelectedObjects() )
                    object->setGlobalVisibility( true );

                for ( auto child : object->children() )
                    objects.push( child );
            }
        }
        someChanges = true;
    }

    return someChanges;
}

bool RibbonMenu::drawCloneSelectionButton( const std::vector<std::shared_ptr<Object>>& selected )
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

bool RibbonMenu::drawMergeSubtreeButton( const std::vector<std::shared_ptr<Object>>& selected )
{
    std::vector<TypedFlatTree> subtrees;
    for ( const auto& subtree : getFlatSubtrees( selected ) )
        subtrees.emplace_back( TypedFlatTree::fromFlatTree( subtree ) );
    if ( subtrees.empty() )
        return false;

    bool needToMerge = false;
    for ( const auto& subtree : subtrees )
    {
        const auto& rootObj = subtree.root;
        needToMerge = needToMerge
            || ( subtree.objsMesh.size() + int( rootObj->asType<ObjectMesh>() != nullptr ) > 1 )
            || ( subtree.objsLines.size() + int( rootObj->asType<ObjectLines>() != nullptr ) > 1 )
            || ( subtree.objsPoints.size() + int( rootObj->asType<ObjectPoints>() != nullptr ) > 1 );
    }
    if ( !needToMerge )
        return false;

    if ( !UI::button( "Combine Subtree", Vector2f( -1, 0 ) ) )
        return false;

    SCOPED_HISTORY( "Combine Subtree" );
    for ( auto& subtree : subtrees )
        mergeSubtree( std::move( subtree ) );

    return true;
}

void RibbonMenu::drawBigButtonItem_( const MenuItemInfo& item )
{
    auto width = buttonDrawer_.calcItemWidth( item, DrawButtonParams::SizeType::Big );

    auto availReg = ImGui::GetContentRegionAvail();

    const auto& style = ImGui::GetStyle();
    ImVec2 itemSize = ImVec2( width.baseWidth, availReg.y - 2 * style.WindowPadding.y );

    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + availReg.y * 0.5f - itemSize.y * 0.5f );

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
                                                              const std::string& tabName, bool centerItems )
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
            if ( ( res[i].numBig + res[i].numSmallText > 0 ) && ( res[i].numBig + res[i].numSmallText + res[i].numSmall != 1 ) )
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

    if ( centerItems && sumWidth < screenWidth )
        ImGui::SetCursorPosX( 0.5f * ( screenWidth - sumWidth ) );

    return res;
}

void RibbonMenu::setupItemsGroup_( const std::vector<std::string>& groupsInTab, const std::string& tabName, bool centerItems )
{
    for ( const auto& g : groupsInTab )
    {
        ImGui::TableSetupColumn( ( g + "##" + tabName ).c_str(), 0 );
    }
    if ( !centerItems )
        ImGui::TableSetupColumn( ( "##fictiveGroup" + tabName ).c_str(), 0 );
}

void RibbonMenu::drawItemsGroup_( const std::string& tabName, const std::string& groupName,
                                  DrawGroupConfig config ) // copy here for easier usage
{
    auto itemSpacing = ImGui::GetStyle().ItemSpacing;
    itemSpacing.y = 3.0f * menu_scaling();
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

bool RibbonMenu::itemPressed_( const std::shared_ptr<RibbonMenuItem>& item, const std::string& requiremetnsHint )
{
    bool available = requiremetnsHint.empty();
    bool wasActive = item->isActive();
    // take name before, because item can become invalid during `action`
    auto name = item->name();
    if ( !wasActive && available && ( activeBlockingItem_.item && item->blocking() ) )
    {
        const auto blockingItemName = activeBlockingItem_.item->name();
        bool closed = true;
        if ( autoCloseBlockingPlugins_ )
            closed = activeBlockingItem_.item->action();

        if ( !closed )
        {
            blockingHighlightTimer_ = 2.0f;
            pushNotification( {
                .text = "Unable to close this plugin",
                .type = NotificationType::Warning } );
            return false;
        }

        if ( !autoCloseBlockingPlugins_ )
        {
            blockingHighlightTimer_ = 2.0f;
            spdlog::info( "Cannot activate item: \"{}\", Active: \"{}\"", name, blockingItemName );
            static bool alreadyShown = false;
            if ( alreadyShown )
                return false;

            alreadyShown = true;
            pushNotification( {
                .onButtonClick = []
                {
                    auto viewerSettingsIt = RibbonSchemaHolder::schema().items.find( "Viewer settings" );
                    if ( viewerSettingsIt == RibbonSchemaHolder::schema().items.end() )
                        return;
                    if ( viewerSettingsIt->second.item && !viewerSettingsIt->second.item->isActive() )
                        viewerSettingsIt->second.item->action();
                },
                .buttonName = "Open Settings",
                .text = "Unable to activate this tool because another blocking tool is already active.\nIt can be changed in the Settings.",
                .type = NotificationType::Info } );
            return false;
        }
        else
        {
            static bool alreadyShown = false;
            spdlog::info( "Activated item: \"{}\", Closed item: \"{}\"", name, blockingItemName );
            if ( !alreadyShown )
            {
                alreadyShown = true;

                pushNotification( {
                .onButtonClick = []
                {
                    auto viewerSettingsIt = RibbonSchemaHolder::schema().items.find( "Viewer settings" );
                    if ( viewerSettingsIt == RibbonSchemaHolder::schema().items.end() )
                        return;
                    if ( viewerSettingsIt->second.item && !viewerSettingsIt->second.item->isActive() )
                        viewerSettingsIt->second.item->action();
                },
                .buttonName = "Open Settings",
                .text = "That tool was closed due to other tool start.\nIt can be changed in the Settings.",
                .type = NotificationType::Info } );
            }
        }
    }
    if ( !wasActive && !available )
    {
        if ( !requiremetnsHint.empty() )
            showModal( requiremetnsHint, NotificationType::Info );
        return false;
    }
    ImGui::CloseCurrentPopup();
    int conflicts = getViewerInstance().mouseController().getMouseConflicts();
    bool stateChanged = item->action();
    if ( !stateChanged )
        spdlog::info( "Action item: \"{}\"", name );
    else
        spdlog::info( "{} item: \"{}\"", wasActive ? std::string( "Deactivated" ) : std::string( "Activated" ), name );

    if ( !wasActive )
    {
        if ( stateChanged && getViewerInstance().mouseController().getMouseConflicts() > conflicts )
        {
            pushNotification( {
                .text = "Camera operations that are controlled by left mouse button "
                        "may not work while this tool is active\n"
                        "Hold Alt additionally to control camera",
                .type = NotificationType::Info,
                .lifeTimeSec = 3.0f } );
        }
    }
    return true;
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
    return item->isAvailable( SceneCache::getAllObjects<const Object, ObjectSelectivityType::Selected>() );
}

void RibbonMenu::drawSceneListButtons_()
{
    auto menuScaling = menu_scaling();
    const float size = ( cMiddleIconSize + 9.f ) * menuScaling;
    const ImVec2 smallItemSize = { size, size };

    const DrawButtonParams params{ DrawButtonParams::SizeType::Small, smallItemSize, cMiddleIconSize,DrawButtonParams::RootType::Toolbar };

    ImGui::SetCursorPosY( ImGui::GetCursorPosY() - 2 * menuScaling );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 6 * menuScaling, 5 * menuScaling ) );
    auto font = fontManager_.getFontByType( RibbonFontManager::FontType::Small );
    //font->Scale = 0.75f;
    ImGui::PushFont( font );
    UI::TestEngine::pushTree( "RibbonSceneButtons" );
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
    UI::TestEngine::popTree(); // "RibbonSceneButtons"
    ImGui::NewLine();
    ImGui::PopFont();
    const float separateLinePos = ImGui::GetCursorScreenPos().y;

    ImGui::PopStyleVar();
    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddLine( ImVec2( 0, separateLinePos ), ImVec2( float( sceneSize_.x ), separateLinePos ),
                                                                  ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Borders ).getUInt32() );
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ImGui::GetStyle().ItemSpacing.y + 1.0f * menuScaling );
}

void RibbonMenu::readMenuItemsStructure_()
{
    MR_TIMER;
    RibbonSchemaLoader loader;
    loader.loadSchema();
    toolbar_->resetItemsList();
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
    toolbar_->setScaling( menu_scaling() );
    fixViewportsSize_( Viewer::instanceRef().framebufferSize.x, Viewer::instanceRef().framebufferSize.y );

    RibbonSchemaLoader loader;
    loader.recalcItemSizes();
}

void RibbonMenu::drawItemDialog_( DialogItemPtr& itemPtr )
{
    if ( !itemPtr.item )
        return;

    auto statePlugin = std::dynamic_pointer_cast< StateBasePlugin >( itemPtr.item );
    if ( !statePlugin || !statePlugin->isEnabled() )
        return;
    statePlugin->preDrawUpdate();

    // check before drawDialog to avoid calling something like:
    // ImGui::Image( textId ) // with removed texture in deferred render calls
    if ( !statePlugin->dialogIsOpen() )
    {
        itemPressed_( itemPtr.item );
        if ( !itemPtr.item )
            return; // do not proceed if we closed dialog in this call
    }

    statePlugin->drawDialog( menu_scaling(), ImGui::GetCurrentContext() );

    if ( !itemPtr.item ) // if it was closed in drawDialog
        return;

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

    if ( !statePlugin->dialogIsOpen() ) // still need to check here we ordered to close dialog in `drawDialog`
        itemPressed_( itemPtr.item );
    else if ( prevFrameSelectedObjectsCache_ != SceneCache::getAllObjects<const Object, ObjectSelectivityType::Selected>() )
        statePlugin->updateSelection( SceneCache::getAllObjects<const Object, ObjectSelectivityType::Selected>() );
}

void RibbonMenu::drawRibbonSceneList_()
{
    const auto& selectedObjs = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();

    const auto scaling = menu_scaling();
    // Define next window position + size
    auto& viewerRef = Viewer::instanceRef();

    float topShift = 0.0f;
    const bool hasTopPanel = menuUIConfig_.topLayout != RibbonTopPanelLayoutMode::None;
    if ( hasTopPanel )
        topShift = float( currentTopPanelHeight_ );

    ImGui::SetWindowPos( "RibbonScene", ImVec2( 0.f, topShift * scaling - 1 ), ImGuiCond_Always );
    const float cMinSceneWidth = 100 * scaling;
    const float cMaxSceneWidth = std::max( cMinSceneWidth, std::round( viewerRef.framebufferSize.x * 0.5f ) );
    sceneSize_.x = std::max( sceneSize_.x, cMinSceneWidth );
    sceneSize_.y = std::round( viewerRef.framebufferSize.y - ( topShift - 2.0f ) * scaling );
    ImGui::SetWindowSize( "RibbonScene", sceneSize_, ImGuiCond_Always );
    ImGui::SetNextWindowSizeConstraints( ImVec2( cMinSceneWidth, -1.f ), ImVec2( cMaxSceneWidth, -1.f ) ); // TODO take out limits to special place
    ImGui::PushStyleVar( ImGuiStyleVar_Alpha, 1.f );
    auto colorBg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    colorBg.w = 1.f;
    ImGui::PushStyleColor( ImGuiCol_WindowBg, colorBg );

    ImGui::Begin(
        "RibbonScene", nullptr,
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoResize
    );
    if ( hasTopPanel )
        drawSceneListButtons_();
    sceneObjectsList_->draw( -( informationHeight_ + transformHeight_ ), menu_scaling() );
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

void RibbonMenu::drawRibbonSceneInformation_( const std::vector<std::shared_ptr<Object>>& /*selected*/ )
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

bool RibbonMenu::drawCollapsingHeaderTransform_()
{
    auto res = drawCollapsingHeader_( "Transform", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowOverlap );

    const float scaling = menu_scaling();
    ImVec2 smallBtnSize = ImVec2( 22 * scaling, 22 * scaling );
    float numButtons = ( sceneSize_.x - 100 * scaling - ImGui::GetStyle().WindowPadding.x * 0.5f ) / smallBtnSize.x;
    if ( numButtons < 1.0f )
        return res;

    auto startPos = ImGui::GetCursorPos();
    auto contextBtnPos = startPos;
    contextBtnPos.x += ( ImGui::GetContentRegionAvail().x + ImGui::GetStyle().WindowPadding.x * 0.5f - smallBtnSize.x );
    contextBtnPos.y += ( -ImGui::GetFrameHeightWithSpacing() + ( ImGui::GetFrameHeight() - smallBtnSize.y ) * 0.5f );

    ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0 );

    auto iconsFont = fontManager_.getFontByType( RibbonFontManager::FontType::Icons );
    if ( iconsFont )
    {
        iconsFont->Scale = 12.f / fontManager_.getFontSizeByType( RibbonFontManager::FontType::Icons );
        ImGui::PushFont( iconsFont );
    }

    ImGui::SetCursorPos( contextBtnPos );

    if ( ImGui::Button( "\xef\x85\x82", smallBtnSize ) ) // three dots icon to open context dialog
        ImGui::OpenPopup( cTransformContextName );
    if ( iconsFont )
        ImGui::PopFont();
    UI::setTooltipIfHovered( "Open Transform Data context menu.", scaling );
    if ( iconsFont )
        ImGui::PushFont( iconsFont );

    const auto& selectedObjectsCache = SceneCache::getAllObjects<const Object, ObjectSelectivityType::Selected>();
    if ( numButtons >= 2.0f && selectedObjectsCache.size() == 1 && selectedObjectsCache.front()->xf() != AffineXf3f() )
    {
        auto obj = std::const_pointer_cast< Object >( selectedObjectsCache.front() );
        assert( obj );
        contextBtnPos.x -= smallBtnSize.x;
        ImGui::SetCursorPos( contextBtnPos );

        if ( ImGui::Button( "\xef\x80\x8d", smallBtnSize ) ) // X(cross) icon for reset
        {
            AppendHistory<ChangeXfAction>( "Reset Transform", obj );
            obj->setXf( AffineXf3f() );
        }
        if ( iconsFont )
            ImGui::PopFont();
        UI::setTooltipIfHovered( "Resets transform value to identity.", scaling );
        if ( iconsFont )
            ImGui::PushFont( iconsFont );

        auto item = RibbonSchemaHolder::schema().items.find( "Apply Transform" );
        bool drawApplyBtn = numButtons >=3.0f &&
            item != RibbonSchemaHolder::schema().items.end() &&
            item->second.item->isAvailable( selectedObjectsCache ).empty();

        if ( drawApplyBtn )
        {
            contextBtnPos.x -= smallBtnSize.x;
            ImGui::SetCursorPos( contextBtnPos );

            if ( ImGui::Button( "\xef\x80\x8c", smallBtnSize ) ) // V(apply) icon for apply
                item->second.item->action();
            if ( iconsFont )
                ImGui::PopFont();
            UI::setTooltipIfHovered( "Transforms object and resets transform value to identity.", scaling );
            if ( iconsFont )
                ImGui::PushFont( iconsFont );
        }
    }
    if ( iconsFont )
    {
        ImGui::PopFont();
        iconsFont->Scale = 1.0f;
    }
    ImGui::PopStyleColor( 3 );
    ImGui::PopStyleVar();
    return res;
}

bool RibbonMenu::drawTransformContextMenu_( const std::shared_ptr<Object>& selected )
{
    if ( !ImGui::BeginPopupContextItem( cTransformContextName ) )
        return false;

    const float scaling = menu_scaling();
    auto buttonSize = 100.0f * scaling;

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

    auto semiBoldFont = fontManager_.getFontByType( RibbonFontManager::FontType::SemiBold );
    if ( semiBoldFont )
        ImGui::PushFont( semiBoldFont );
    ImGui::Text( "Transform Data" );
    if ( semiBoldFont )
        ImGui::PopFont();

    const auto& startXf = selected->xf();
#if !defined( __EMSCRIPTEN__ )
    if ( UI::button( "Copy", Vector2f( buttonSize, 0 ) ) )
    {
        Json::Value root;
        serializeTransform( root, { startXf, uniformScale_ } );
        transformClipboardText_ = root.toStyledString();
        if ( auto res = SetClipboardText( transformClipboardText_ ); !res )
            spdlog::warn( res.error() );
        ImGui::CloseCurrentPopup();
    }
#endif
    if ( ImGui::IsWindowAppearing() )
    {
        if ( auto text = GetClipboardText() )
            transformClipboardText_ = *text;
        else
            spdlog::warn( text.error() );
    }

    if ( !transformClipboardText_.empty() )
    {
        if ( auto root = deserializeJsonValue( transformClipboardText_ ) )
        {
            if ( auto tr = deserializeTransform( *root ) )
            {
                if ( UI::button( "Paste", Vector2f( buttonSize, 0 ) ) )
                {
                    AppendHistory<ChangeXfAction>( "Paste Transform", selected );
                    selected->setXf( tr->xf );
                    uniformScale_ = tr->uniformScale;
                    ImGui::CloseCurrentPopup();
                }
            }
        }
    }

    if ( UI::button( "Save to file", Vector2f( buttonSize, 0 ) ) )
    {
        auto filename = saveFileDialog( {
            .fileName = "Transform",
            .filters = { {"JSON (.json)", "*.json"} },
        } );
        if ( !filename.empty() )
        {
            Json::Value root;
            serializeTransform( root, { startXf, uniformScale_ } );

            // although json is a textual format, we open the file in binary mode to get exactly the same result on Windows and Linux
            std::ofstream ofs( filename, std::ofstream::binary );
            if ( ofs )
                ofs << root.toStyledString();
            else
                spdlog::error( "Cannot open file for writing" );
        }
        ImGui::CloseCurrentPopup();
    }

    if ( UI::button( "Load from file", Vector2f( buttonSize, 0 ) ) )
    {
        auto filename = openFileDialog( { .filters = { { "JSON (.json)", "*.json" } } } );
        if ( !filename.empty() )
        {
            std::string errorString;
            if ( auto root = deserializeJsonValue( filename ) )
            {
                if ( auto tr = deserializeTransform( *root ) )
                {
                    AppendHistory<ChangeXfAction>( "Load Transform from File", selected );
                    selected->setXf( tr->xf );
                    uniformScale_ = tr->uniformScale;
                }
                else
                {
                    errorString = "Cannot parse transform";
                }
            }
            else
            {
                errorString = "Cannot parse transform";
            }
            if ( !errorString.empty() )
                pushNotification( { .text = errorString, .type = NotificationType::Error } );
        }
        ImGui::CloseCurrentPopup();
    }

    if ( startXf != AffineXf3f() )
    {
        auto item = RibbonSchemaHolder::schema().items.find( "Apply Transform" );
        if ( item != RibbonSchemaHolder::schema().items.end() &&
            item->second.item->isAvailable( SceneCache::getAllObjects<const Object, ObjectSelectivityType::Selected>() ).empty() &&
            UI::button( "Apply", Vector2f( buttonSize, 0 ) ) )
        {
            item->second.item->action();
            ImGui::CloseCurrentPopup();
        }
        UI::setTooltipIfHovered( "Transforms object and resets transform value to identity.", scaling );

        if ( UI::button( "Reset", Vector2f( buttonSize, 0 ) ) )
        {
            AppendHistory<ChangeXfAction>( "Reset Transform (context menu)", selected );
            selected->setXf( AffineXf3f() );
            ImGui::CloseCurrentPopup();
        }
        UI::setTooltipIfHovered( "Resets transform value to identity.", scaling );
    }
    ImGui::EndPopup();
    return true;
}

void RibbonMenu::addRibbonItemShortcut_( const std::string& itemName, const ShortcutManager::ShortcutKey& key, ShortcutManager::Category category )
{
    if ( !shortcutManager_ )
    {
        assert( false );
        return;
    }
    auto itemIt = RibbonSchemaHolder::schema().items.find( itemName );
    if ( itemIt != RibbonSchemaHolder::schema().items.end() )
    {
        shortcutManager_->setShortcut( key, { category, itemIt->first, [item = itemIt->second.item, this]()
        {
            itemPressed_( item, getRequirements_( item ) );
        } } );
    }
#ifndef __EMSCRIPTEN__
    else
    {
        spdlog::error( "Ribbon item not found: {}", itemName );
        assert( !"item not found" );
    }
#endif
}

void RibbonMenu::setupShortcuts_()
{
    ImGuiMenu::setupShortcuts_();
    if ( !shortcutManager_ )
    {
        assert( false );
        return;
    }

    shortcutManager_->setShortcut( { GLFW_KEY_H,0 }, { ShortcutManager::Category::View, "Toggle selected objects visibility", [] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto& selected = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();
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
        const auto& selected = SceneCache::getAllObjects<ObjectMeshHolder, ObjectSelectivityType::Selected>();
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( MeshVisualizePropertyType::FlatShading, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_F, CONTROL_OR_SUPER }, { ShortcutManager::Category::Info, "Search plugin by name or description",[this] ()
    {
        searcher_.activate();
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_I,0 }, { ShortcutManager::Category::View, "Invert normals of selected objects",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto& selected = SceneCache::getAllObjects<VisualObject, ObjectSelectivityType::Selected>();
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( VisualizeMaskType::InvertedNormals, viewportid );
    } }  );
    shortcutManager_->setShortcut( { GLFW_KEY_L,0 }, { ShortcutManager::Category::View, "Toggle edges on selected meshes",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto& selected = SceneCache::getAllObjects<ObjectMeshHolder, ObjectSelectivityType::Selected>();
        for ( const auto& sel : selected )
                sel->toggleVisualizeProperty( MeshVisualizePropertyType::Edges, viewportid );
    } } );
    shortcutManager_->setShortcut( { GLFW_KEY_KP_5,0 }, { ShortcutManager::Category::View, "Toggle Orthographic/Perspective View",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        viewport.setOrthographic( !viewport.getParameters().orthographic );
    } }  );
    shortcutManager_->setShortcut( { GLFW_KEY_T,0 }, { ShortcutManager::Category::View, "Toggle faces on selected meshes",[] ()
    {
        auto& viewport = getViewerInstance().viewport();
        const auto& viewportid = viewport.id;
        const auto& selected = SceneCache::getAllObjects<ObjectMeshHolder, ObjectSelectivityType::Selected>();
        for ( const auto& sel : selected )
            sel->toggleVisualizeProperty( MeshVisualizePropertyType::Faces, viewportid );
    } }  );
    if ( sceneObjectsList_ )
    {
        shortcutManager_->setShortcut( { GLFW_KEY_DOWN,0 }, { ShortcutManager::Category::Objects, "Select next object",[&] ()
        {
            sceneObjectsList_->changeSelection( true, false );
        } } );
        shortcutManager_->setShortcut( { GLFW_KEY_DOWN,GLFW_MOD_SHIFT }, { ShortcutManager::Category::Objects, "Add next object to selection",[&] ()
        {
            sceneObjectsList_->changeSelection( true, true );
        } } );
        shortcutManager_->setShortcut( { GLFW_KEY_UP,0 }, { ShortcutManager::Category::Objects, "Select previous object",[&] ()
        {
            sceneObjectsList_->changeSelection( false, false );
        } } );
        shortcutManager_->setShortcut( { GLFW_KEY_UP,GLFW_MOD_SHIFT }, { ShortcutManager::Category::Objects, "Add previous object to selection",[&] ()
        {
            sceneObjectsList_->changeSelection( false, true );
        } } );
        shortcutManager_->setShortcut( { GLFW_KEY_A, CONTROL_OR_SUPER }, { ShortcutManager::Category::Objects, "Ribbon Scene Select all",[&] ()
        {
            sceneObjectsList_->selectAllObjects();
        } } );
        shortcutManager_->setShortcut( { GLFW_KEY_F3, 0 }, { ShortcutManager::Category::View, "Ribbon Scene Show only previous",[&] ()
        {
            sceneObjectsList_->changeVisible( false );
        } } );
        shortcutManager_->setShortcut( { GLFW_KEY_F4, 0 }, { ShortcutManager::Category::View, "Ribbon Scene Show only next",[&] ()
        {
            sceneObjectsList_->changeVisible( true );
        } } );
    }

    addRibbonItemShortcut_( "Fit data", { GLFW_KEY_F, GLFW_MOD_CONTROL | GLFW_MOD_ALT }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Top View", { GLFW_KEY_KP_7, 0 }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Front View", { GLFW_KEY_KP_1, 0 }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Right View", { GLFW_KEY_KP_3, 0 }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Invert View", { GLFW_KEY_KP_9, 0 }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Bottom View", { GLFW_KEY_KP_7, CONTROL_OR_SUPER }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Back View", { GLFW_KEY_KP_1, CONTROL_OR_SUPER }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Left View", { GLFW_KEY_KP_3, CONTROL_OR_SUPER }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Show_Hide Global Basis", { GLFW_KEY_G, CONTROL_OR_SUPER }, ShortcutManager::Category::View );
    addRibbonItemShortcut_( "Select objects", { GLFW_KEY_Q, GLFW_MOD_CONTROL }, ShortcutManager::Category::Objects );
    addRibbonItemShortcut_( "Open files", { GLFW_KEY_O, CONTROL_OR_SUPER }, ShortcutManager::Category::Scene );
    addRibbonItemShortcut_( "Save Scene", { GLFW_KEY_S, CONTROL_OR_SUPER }, ShortcutManager::Category::Scene );
    addRibbonItemShortcut_( "Save Scene As", { GLFW_KEY_S, CONTROL_OR_SUPER | GLFW_MOD_SHIFT }, ShortcutManager::Category::Scene );
    addRibbonItemShortcut_( "New", { GLFW_KEY_N, CONTROL_OR_SUPER }, ShortcutManager::Category::Scene );
    addRibbonItemShortcut_( "Ribbon Scene Rename", { GLFW_KEY_F2, 0 }, ShortcutManager::Category::Objects );
    addRibbonItemShortcut_( "Ribbon Scene Remove selected objects", { GLFW_KEY_R, GLFW_MOD_SHIFT }, ShortcutManager::Category::Objects );
    addRibbonItemShortcut_( "Viewer settings", { GLFW_KEY_COMMA, CONTROL_OR_SUPER }, ShortcutManager::Category::Info );
}

void RibbonMenu::drawShortcutsWindow_()
{
    if ( !shortcutManager_ )
    {
        assert( false );
        return;
    }

    const auto& style = ImGui::GetStyle();
    const auto scaling = menu_scaling();
    float windowWidth = 1000.0f * scaling;

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
    windowHeight += 40.0f * scaling; // Reserve a bit more space

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
        UI::inputText( ( "##" + line + std::to_string( ++lineIndexer ) ).c_str(), const_cast< std::string& >( line ), ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_AutoSelectAll );
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
            const auto& [key, category, name] = shortcutList[i];
            const auto& caption = getItemCaption( name );

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
            ImGui::Text( "%s", caption.c_str() );
            ImGui::PopStyleColor();

            float textSize = ImGui::CalcTextSize( caption.c_str() ).x;
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

            if ( key.mod & GLFW_MOD_SUPER )
            {
                ImGui::SetCursorPosY( ImGui::GetCursorPosY() - cButtonPadding * scaling );
                addReadOnlyLine( ShortcutManager::getModifierString( GLFW_MOD_SUPER ) );
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

void RibbonMenu::updateTopPanelSize_( bool drawTabs )
{
    constexpr int cTabSize = int( cTabHeight + cTabYOffset + 1 );
    constexpr int cSumPanelSizeSize = cTabSize + 80;
    if ( drawTabs && topPanelHiddenHeight_ == cTabSize )
        return;
    if ( !drawTabs && topPanelHiddenHeight_ == 0 )
        return;
    if ( drawTabs )
    {
        topPanelOpenedHeight_ = cSumPanelSizeSize;
        topPanelHiddenHeight_ = cTabSize;
    }
    else
    {
        topPanelOpenedHeight_ = cSumPanelSizeSize - cTabSize;
        topPanelHiddenHeight_ = 0;
        collapseState_ = CollapseState::Pinned;
    }
    currentTopPanelHeight_ = collapseState_ == CollapseState::Closed ? topPanelHiddenHeight_ : topPanelOpenedHeight_;

    fixViewportsSize_( getViewerInstance().framebufferSize.x, getViewerInstance().framebufferSize.y );
}

void RibbonMenu::drawTopPanel_( bool drawTabs, bool centerItems )
{
    updateTopPanelSize_( drawTabs );

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
    drawTopPanelOpened_( drawTabs, centerItems );
}

void RibbonMenu::drawTopPanelOpened_( bool drawTabs, bool centerItems )
{
    beginTopPanel_();

    const auto& style = ImGui::GetStyle();
    auto itemSpacing = style.ItemSpacing;
    itemSpacing.x = cRibbonItemInterval * menu_scaling();
    auto cellPadding = style.CellPadding;
    cellPadding.x = itemSpacing.x;
    auto framePadding = style.FramePadding;
    framePadding.x = 0.0f;

    if ( drawTabs )
        drawHeaderPannel_();

    // tab content position
    ImGui::SetCursorPosX( style.CellPadding.x * 2 );
    if ( drawTabs )
        ImGui::SetCursorPosY( ( cTabYOffset + cTabHeight ) * menu_scaling() + 2 );
    else
        ImGui::SetCursorPosY( 2 );

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
            auto config = setupItemsGroupConfig_( tabIt->second, tab, centerItems );
            if ( ImGui::BeginTable( ( tab + "##table" ).c_str(), int( tabIt->second.size() ) + ( centerItems ? 0 : 1 ), tableFlags ) )
            {
                setupItemsGroup_( tabIt->second, tab, centerItems );
                ImGui::TableNextRow();
                UI::TestEngine::pushTree( "Ribbon" );
                for ( int i = 0; i < tabIt->second.size(); ++i )
                {
                    const auto& group = tabIt->second[i];
                    ImGui::TableNextColumn();
                    drawItemsGroup_( tab, group, config[i] );
                }
                UI::TestEngine::popTree(); // "Ribbon"
                if ( !centerItems )
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
    if ( width == 0 || height == 0 )
        return;
    auto viewportsBounds = viewer->getViewportsBounds();
    auto minMaxDiff = viewportsBounds.max - viewportsBounds.min;

    float topPanelHeightScaled = 0.0f;
    if ( menuUIConfig_.topLayout != RibbonTopPanelLayoutMode::None )
    {
        topPanelHeightScaled =
            ( collapseState_ == CollapseState::Pinned ? topPanelOpenedHeight_ : topPanelHiddenHeight_ ) *
            menu_scaling();
    }
    for ( auto& vp : viewer->viewport_list )
    {
        auto rect = vp.getViewportRect();

        float sceneWidth = 0.0f;
        if ( menuUIConfig_.drawScenePanel )
            sceneWidth = sceneSize_.x;

        auto widthRect = MR::width( rect );
        auto heightRect = MR::height( rect );

        // -2 - buffer pixel
        rect.min.x = ( rect.min.x - viewportsBounds.min.x ) / minMaxDiff.x * ( width - ( sceneWidth - 2 ) ) + sceneWidth;
        rect.min.y = ( rect.min.y - viewportsBounds.min.y ) / minMaxDiff.y * ( height - ( topPanelHeightScaled - 2 ) );
        rect.max.x = rect.min.x + widthRect / minMaxDiff.x * ( width - ( sceneWidth - 2 ) );
        rect.max.y = rect.min.y + heightRect / minMaxDiff.y * ( height - ( topPanelHeightScaled - 2 ) );
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
        // Fix ImGui.
        // The logic is set inside the library that if the program got there,
        // then the command buffer should be at least one, but possibly empty.
        // there is a blocking window that is not currently displayed.
        // at some point, when trying to open another window, a crash occurs
        // (it is worth switching applications during playback so that the system makes the
        // window of another application active).
        if ( drawList->CmdBuffer.Size > 0)
        {
            drawList->PushClipRect( ImVec2( 0, 0 ), ImGui::GetIO().DisplaySize );
            drawList->AddRect(
                ImVec2( window->Pos.x - 2.0f * scaling, window->Pos.y - 2.0f * scaling ),
                ImVec2( window->Pos.x + window->Size.x + 2.0f * scaling, window->Pos.y + window->Size.y + 2.0f * scaling ),
                highlightColor.getUInt32(), 0.0f, 0, 2.0f * scaling );
            drawList->PopClipRect();
        }
    }
    getViewerInstance().incrementForceRedrawFrames();
    blockingHighlightTimer_ -= ImGui::GetIO().DeltaTime;
}

void pushNotification( const RibbonNotification& notification )
{
    if ( auto ribbonMenu = getViewerInstance().getMenuPluginAs<RibbonMenu>() )
    {
        if ( notification.text.back() != '\n' )
            return ribbonMenu->pushNotification( notification );

        auto notificationCopy = notification;
        notificationCopy.text.pop_back();
        return ribbonMenu->pushNotification( notificationCopy );
    }

    showModal( notification.text, notification.type );
}

}
