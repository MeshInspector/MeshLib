#pragma once
#include "MRMenu.h"
#include "MRRibbonMenuItem.h"
#include "MRRibbonFontManager.h"
#include "MRRibbonButtonDrawer.h"
#include "MRAsyncTimer.h"
#include "MRRibbonSchema.h"
#include "MRShortcutManager.h"
#include <boost/signals2/signal.hpp>
#include <type_traits>
#include <array>

namespace MR
{

class Object;


// Class to control and render ribbon-style menu
// stores menu items,
// menu structure is provided by `menuItemsStructure.json` file (parsed on init)
class MRVIEWER_CLASS RibbonMenu : public ImGuiMenu
{
public:
    MRVIEWER_API RibbonMenu();
    MRVIEWER_API virtual ~RibbonMenu();

    MRVIEWER_API virtual void init( MR::Viewer* _viewer ) override;

    MRVIEWER_API virtual void shutdown() override;

    // get access to quick access menu items list
    MenuItemsList& getQuickAccessList() { return toolbarItemsList_; };
    /// get maximum items in quick access menu
    MRVIEWER_API int getToolbarMaxItemCount() const;
    /// open Toolbar Customize modal popup
    MRVIEWER_API void openToolbarCustomize();

    MRVIEWER_API virtual void load_font( int font_size = 13 ) override;

    int getTopPanelHeight() const
    {
        return currentTopPanelHeight_;
    };

    MRVIEWER_API virtual std::filesystem::path getMenuFontPath() const override;

    // set top panel pinned and unpinned
    MRVIEWER_API virtual void pinTopPanel( bool on );
    MRVIEWER_API bool isTopPannelPinned() const;
    // this functions allow you to get top panel height (before scaling)
    int getTopPanelOpenedHeight() const { return topPanelOpenedHeight_; }
    int getTopPanelHiddenHeight() const { return topPanelHiddenHeight_; }
    int getTopPanelCurrentHeight() const { return currentTopPanelHeight_; }
    // set maximum wait time (in seconds) before top panel is closed after mouse leaves it (only when not pinned)
    // minimum value is 0 seconds, panel will close immediately after mouse leaves it
    void setTopPanelMaxOpenedTimer( float sec ) { openedMaxSecs_ = std::max( 0.0f, sec ); }

    /// read quick access menu items list from json
    MRVIEWER_API virtual void readQuickAccessList( const Json::Value& root );

    /// reset quick access menu items list to default
    MRVIEWER_API void resetQuickAccessList();

    /// get access to Ribbon font manager
    const RibbonFontManager& getFontManager() { return fontManager_; };

    /// get Scene List window size
    Vector2i getSceneSize() { return Vector2i( int( sceneSize_.x ), int( sceneSize_.y ) ); };

    /// set Scene List window size
    MRVIEWER_API void setSceneSize( const Vector2i& size );

    /// returns true if any blocking plugin is now active
    bool hasActiveBlockingItem() const { return bool( activeBlockingItem_.item ); }

    /// updates status of item if it was changed outside of menu
    MRVIEWER_API void updateItemStatus( const std::string& itemName );

    /// returns index of active tab in RibbonSchemaHolder::schema().tabsOrder
    int getActiveTabIndex() const { return activeTabIndex_; }

    using TabChangedSignal = boost::signals2::signal<void( int prevTabId, int newTabId )>;
    /// this signal is called when active tab changes
    TabChangedSignal tabChangedSignal;

    const RibbonButtonDrawer& getRibbonButtonDrawer() { return buttonDrawer_; }

protected:

    // draw single item
    MRVIEWER_API virtual void drawBigButtonItem_( const MenuItemInfo& item );
    // draw set of small text buttons
    MRVIEWER_API virtual void drawSmallButtonsSet_( const std::vector<std::string>& group, int setFrontIndex, int setLength, 
                                                    bool withText );

    // Configuration of ribbon group
    struct DrawGroupConfig
    {
        int numBig{ 0 };
        int numSmallText{ 0 };
        int numSmall{ 0 };
    };
    using DrawTabConfig = std::vector<DrawGroupConfig>;
    // draw group of items
    MRVIEWER_API virtual DrawTabConfig setupItemsGroupConfig_( const std::vector<std::string>& groupsInTab, const std::string& tabName );
    MRVIEWER_API virtual void setupItemsGroup_( const std::vector<std::string>& groupsInTab, const std::string& tabName );
    MRVIEWER_API virtual void drawItemsGroup_( const std::string& tabName, const std::string& groupName, 
                                               DrawGroupConfig config );
    // ribbon menu item pressed
    MRVIEWER_API virtual void itemPressed_( const std::shared_ptr<RibbonMenuItem>& item, bool available );

    MRVIEWER_API virtual void drawActiveBlockingDialog_();
    MRVIEWER_API virtual void drawActiveNonBlockingDialogs_();

    MRVIEWER_API virtual void postResize_( int width, int height ) override;
    MRVIEWER_API virtual void postRescale_( float x, float y ) override;

    struct DialogItemPtr
    {
        std::shared_ptr<RibbonMenuItem> item;
        // this flag is needed to correctly fix position of UI dialog (temporary, while plugin dialogs are floating in viewport)
        bool dialogPositionFixed{ false };
    };
    DialogItemPtr activeBlockingItem_;
    std::vector<DialogItemPtr> activeNonBlockingItems_;
    MRVIEWER_API virtual void drawItemDialog_( DialogItemPtr& itemPtr );

    // Draw ribbon top panel
    MRVIEWER_API virtual void drawTopPanel_();
    // Draw scene list window with content
    MRVIEWER_API virtual void drawRibbonSceneList_();
    // Draw scene list content only
    MRVIEWER_API virtual void drawRibbonSceneListContent_( std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all );
    // Draw viewport id and projection type for all viewporrts
    MRVIEWER_API virtual void drawRibbonViewportsLabels_();

    MRVIEWER_API virtual void drawRibbonSceneInformation_( std::vector<std::shared_ptr<Object>>& selected );

    MRVIEWER_API virtual void drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& selected ) override;
    MRVIEWER_API virtual bool drawTransformContextMenu_( const std::shared_ptr<Object>& selected ) override;

    MRVIEWER_API virtual void drawToolbarWindow_();
    MRVIEWER_API virtual void drawToolbarCustomizeWindow_();
    MRVIEWER_API virtual void drawToolbarCustomizeTabsList_();
    MRVIEWER_API virtual void drawToolbarCustomizeItemsList_();

    // return icon (now it is symbol in icons font) based on typename
    MRVIEWER_API virtual const char* getSceneItemIconByTypeName_( const std::string& typeName ) const;

    MRVIEWER_API virtual void drawCustomObjectPrefixInScene_( const Object& obj ) override;   

    MRVIEWER_API virtual void addRibbonItemShortcut_( const std::string& itemName, const ShortcutManager::ShortcutKey& key, ShortcutManager::Category category );

    MRVIEWER_API virtual void setupShortcuts_() override;

    MRVIEWER_API virtual void drawShortcutsWindow_() override;
    // reads files with panel description
    MRVIEWER_API virtual void readMenuItemsStructure_();

    std::vector<std::shared_ptr<const Object>> prevFrameObjectsCache_;
    std::vector<std::shared_ptr<const Object>> selectedObjectsCache_;

    MRVIEWER_API virtual bool drawCollapsingHeader_( const char* label, ImGuiTreeNodeFlags flags = 0 ) override;

private:
    void changeTab_( int newTab );

    std::string getRequirements_( const std::shared_ptr<RibbonMenuItem>& item ) const;

    // draw scene list buttons
    void drawSceneListButtons_();

    // struct to hold information for search result presentation
    struct SearchResult
    {
        int tabIndex{ -1 }; // -1 is default value if item has no tab
        const MenuItemInfo* item{ nullptr }; // item info to show correct caption
    };
    // does look up in ribbon schema for `searchLine_`
    std::vector<SearchResult> search_( const std::string& searchStr );
    std::string searchLine_;
    std::vector<SearchResult> searchResult_;
    void drawSearchButton_();
    void drawCollapseButton_();

    void sortObjectsRecursive_( std::shared_ptr<Object> object );

    // part of top panel
    void drawHeaderQuickAccess_();
    void drawHeaderPannel_();
    void drawActiveListButton_( const ImVec2& basePos, float btnSize, float textSize );

    bool drawGroupUngroupButton_( const std::vector<std::shared_ptr<Object>>& selected );

    void beginTopPanel_();
    void endTopPanel_();
    void drawTopPanelOpened_();

    void fixViewportsSize_( int w, int h );

    std::string transformClipboardText_;

    int currentTopPanelHeight_ = 111;
    int topPanelOpenedHeight_ = 111;
    int topPanelHiddenHeight_ = 33;

    ImVec2 sceneSize_{ 310, 0 };
    float informationHeight_{ 0.f };
    float transformHeight_{ 0.f };

    // current scroll position of tabs panel
    float tabPanelScroll_{ 0.0f };

    enum class CollapseState
    {
        Closed,
        Opened,
        Pinned
    } collapseState_{ CollapseState::Pinned };
    // seconds to stay opened if not pinned
    float openedMaxSecs_{ 2.0f };
    float openedTimer_{ openedMaxSecs_ };

    int activeTabIndex_{ 0 };
    RibbonFontManager fontManager_;
    RibbonButtonDrawer buttonDrawer_;

    AsyncTimer asyncTimer_;
    std::thread timerThread_;

    MenuItemsList toolbarItemsList_; // toolbar items list
    MenuItemsList toolbarListCustomize_; // toolbar preview items list for Toolbar Customize window
    bool toolbarDragDrop_ = false; // active drag&drop in Toolbar Customize window
    bool openToolbarCustomizeFlag_ = false; // flag to open Toolbar Customize window
    int toolbarCustomizeTabNum_ = 0;
    std::string toolbarSearch_;
    std::vector<std::vector<std::string>> toolbarSearchRes_;
};

template<typename T>
struct RibbonMenuItemAdder
{
    RibbonMenuItemAdder()
    {
        static_assert( std::is_base_of_v<RibbonMenuItem, T> );
        RibbonSchemaHolder::addItem( std::make_shared<T>() );
    }
};

template<typename T>
struct RibbonMenuItemCall
{
    template<typename Func>
    RibbonMenuItemCall( Func f )
    {
        static_assert( std::is_base_of_v<RibbonMenuItem, T> );
        const auto& items = RibbonSchemaHolder::schema().items;
        for ( const auto& item : items )
        {
            auto plugin = std::dynamic_pointer_cast<T>( item.second.item );
            if ( !plugin )
                continue;
            f( plugin );
        }
    }
};

#define MR_REGISTER_RIBBON_ITEM(pluginType) \
    static MR::RibbonMenuItemAdder<pluginType> ribbonMenuItemAdder##pluginType##_;

#define MR_RIBBON_ITEM_CALL(pluginType,func) \
    static MR::RibbonMenuItemCall<pluginType> ribbonMenuItemCall##func##pluginType##_( func );

}

