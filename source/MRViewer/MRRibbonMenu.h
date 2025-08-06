#pragma once
#include "ImGuiMenu.h"
#include "MRRibbonMenuItem.h"
#include "MRRibbonMenuSearch.h"
#include "MRRibbonFontManager.h"
#include "MRRibbonButtonDrawer.h"
#include "MRAsyncTimer.h"
#include "MRRibbonSchema.h"
#include "MRRibbonMenuUIConfig.h"
#include "MRMesh/MRSignal.h"
#include "MRRibbonNotification.h"
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
    struct CustomContextMenuCheckbox
    {
        using Setter = std::function<void( std::shared_ptr<Object> object, ViewportId id, bool checked )>;
        using Getter = std::function<bool( std::shared_ptr<Object> object, ViewportId id )>;
        Setter setter;
        Getter getter;
        // display a checkBox when only these types of objects are selected
        // by default, it is always hidden
        SelectedTypesMask selectedMask = SelectedTypesMask( -1 );
    };

public:
    MRVIEWER_API RibbonMenu();
    MRVIEWER_API ~RibbonMenu();

    // returns RibonMenu from ViewerInstance()
    MRVIEWER_API static std::shared_ptr<RibbonMenu> instance();

    // adds a custom checkBox to the context menu
    // it is applied to the selected objects
    MRVIEWER_API void setCustomContextCheckbox(
        const std::string& name,
        CustomContextMenuCheckbox customContextMenuCheckbox );

    MRVIEWER_API virtual void init( MR::Viewer* _viewer ) override;

    MRVIEWER_API virtual void shutdown() override;

    /// open Toolbar Customize modal popup
    MRVIEWER_API void openToolbarCustomize();

    MRVIEWER_API virtual void load_font( int font_size = 13 ) override;

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

    /// set quick access menu item list version
    MRVIEWER_API virtual void setQuickAccessListVersion( int version );
    /// read quick access menu items list from json
    MRVIEWER_API virtual void readQuickAccessList( const Json::Value& root );

    /// reset quick access menu items list to default
    MRVIEWER_API void resetQuickAccessList();

    /// get Scene List window size
    Vector2i getSceneSize() { return Vector2i( int( sceneSize_.x ), int( sceneSize_.y ) ); };

    /// set Scene List window size
    MRVIEWER_API void setSceneSize( const Vector2i& size );

    /// returns true if any blocking plugin is now active
    bool hasActiveBlockingItem() const { return bool( activeBlockingItem_.item ); }
    /// returns true if any plugin is now active
    bool hasAnyActiveItem() const { return bool( activeBlockingItem_.item ) || !activeNonBlockingItems_.empty(); }

    /// updates status of item if it was changed outside of menu
    MRVIEWER_API void updateItemStatus( const std::string& itemName );

    /// returns index of active tab in RibbonSchemaHolder::schema().tabsOrder
    int getActiveTabIndex() const { return activeTabIndex_; }


    /// get access to Ribbon font manager
    RibbonFontManager& getFontManager() { return fontManager_; };

    /// get access to Ribbon button drawer
    RibbonButtonDrawer& getRibbonButtonDrawer() { return buttonDrawer_; }

    /// get access to Ribbon Toolbar
    Toolbar& getToolbar() { return *toolbar_; }

    /// get access to Ribbon notifier
    RibbonNotifier& getRibbonNotifier() { return notifier_; };

    void setActiveListPos( const ImVec2& pos ) { activeListPos_ = pos; }
    
    /// set active plugins list showed
    void showActiveList() { activeListPressed_ = true; };

    /// adds new notification to notifier list
    /// draws it first
    MRVIEWER_API virtual void pushNotification( const RibbonNotification& notification );

    /// clones given objects with sub-objects (except for ancillary and unrecognized children) and undo
    MRVIEWER_API static void cloneTree( const std::vector<std::shared_ptr<Object>>& selectedObjects );
    /// clones selected part of given object as separate object (faces, points)
    MRVIEWER_API static void cloneSelectedPart( const std::shared_ptr<Object>& object );

    using TabChangedSignal = boost::signals2::signal<void( int prevTabId, int newTabId )>;
    /// this signal is called when active tab changes
    TabChangedSignal tabChangedSignal;

    /// returns flag defining if closing plugin on opening another one is enabled
    bool getAutoCloseBlockingPlugins() const { return autoCloseBlockingPlugins_; }
    /// sets flag defining if closing plugin on opening another one is enabled or not
    void setAutoCloseBlockingPlugins( bool value ) { autoCloseBlockingPlugins_ = value; }

    /// returns current menu ui configuration (find more in RibbonMenuUIConfig comments)
    const RibbonMenuUIConfig& getMenuUIConfig() const { return menuUIConfig_; }
    MRVIEWER_API virtual void setMenuUIConfig( const RibbonMenuUIConfig& newConfig );

    // ======== selected objects options drawing
    bool drawGroupUngroupButton( const std::vector<std::shared_ptr<Object>>& selected );
    bool drawSelectSubtreeButton( const std::vector<std::shared_ptr<Object>>& selected );
    bool drawCloneButton( const std::vector<std::shared_ptr<Object>>& selected );
    bool drawCustomCheckBox( const std::vector<std::shared_ptr<Object>>& selected, SelectedTypesMask selectedMask );
    bool drawCloneSelectionButton( const std::vector<std::shared_ptr<Object>>& selected );
    bool drawMergeSubtreeButton( const std::vector<std::shared_ptr<Object>>& selected );
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
    MRVIEWER_API virtual DrawTabConfig setupItemsGroupConfig_( const std::vector<std::string>& groupsInTab, const std::string& tabName, bool centerItems );
    MRVIEWER_API virtual void setupItemsGroup_( const std::vector<std::string>& groupsInTab, const std::string& tabName, bool centerItems );
    MRVIEWER_API virtual void drawItemsGroup_( const std::string& tabName, const std::string& groupName,
                                               DrawGroupConfig config );
    // ribbon menu item pressed
    // requiremetnsHint - text that is showed if tool is unavailable (if empty then tool is available)
    // returns true if item was actually activated or deactivated with press action
    MRVIEWER_API virtual bool itemPressed_( const std::shared_ptr<RibbonMenuItem>& item, const std::string& requiremetnsHint = {} );

    /// returns requirements line for given tool, empty line means that tool is available
    MRVIEWER_API virtual std::string getRequirements_( const std::shared_ptr<RibbonMenuItem>& item ) const;

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
    MRVIEWER_API virtual void drawTopPanel_( bool drawTabs = true, bool centerItems = false );
    // Draw scene list window with content
    MRVIEWER_API virtual void drawRibbonSceneList_();
    // Draw vertical line at the right border of scene to enable resize of scene list
    // returns size of scene window
    MRVIEWER_API virtual Vector2f drawRibbonSceneResizeLine_();
    // Draw viewport id and projection type for all viewporrts
    MRVIEWER_API virtual void drawRibbonViewportsLabels_();

    MRVIEWER_API virtual void drawRibbonSceneInformation_( const std::vector<std::shared_ptr<Object>>& selected );

    MRVIEWER_API virtual bool drawCollapsingHeaderTransform_() override;
    MRVIEWER_API virtual bool drawTransformContextMenu_( const std::shared_ptr<Object>& selected ) override;

    MRVIEWER_API virtual void addRibbonItemShortcut_( const std::string& itemName, const ShortcutKey& key, ShortcutCategory category );

    MRVIEWER_API virtual void setupShortcuts_() override;

    MRVIEWER_API virtual void drawShortcutsWindow_() override;
    // reads files with panel description
    MRVIEWER_API virtual void readMenuItemsStructure_();

    std::vector<std::shared_ptr<const Object>> prevFrameSelectedObjectsCache_;

    MRVIEWER_API virtual bool drawCollapsingHeader_( const char* label, ImGuiTreeNodeFlags flags = 0 ) override;

    MRVIEWER_API virtual void highlightBlocking_();

    // draw scene list buttons
    MRVIEWER_API virtual void drawSceneListButtons_();

    // updates viewport sizes with respect to ribbon top and left panels
    MRVIEWER_API virtual void fixViewportsSize_( int w, int h );
    
    // need to be called if you override windows pipeline and use ActiveListPlugin
    MRVIEWER_API void drawActiveList_();

    // call this to draw RibbonNotifier with respect of scene size and ribbon top panel
    MRVIEWER_API virtual void drawNotifications_();

    // this function changes internal sizes of topPanel when it is enabled or disabled
    MRVIEWER_API virtual void updateTopPanelSize_( bool drawTabs );

    // draw quick access bar at header level
    MRVIEWER_API virtual void drawHeaderQuickAccess_( float menuScaling );

    // this functions draws header helpers:
    //  1. Active tools list
    //  2. Search bar
    //  3. Help button
    //  4. Ribbon pin/unpin button
    // returns width available for drawing tabs
    MRVIEWER_API virtual float drawHeaderHelpers_( float requiredTabSize, float menuScaling );

    // helper list of active tools
    MRVIEWER_API virtual void drawActiveListButton_( float btnSize );
    // header helper search bar at panel 
    MRVIEWER_API virtual void drawSearchButton_();
    // header helper button to pin/unpin ribbon
    MRVIEWER_API virtual void drawCollapseButton_();
    // header helper button link to help page
    MRVIEWER_API virtual void drawHelpButton_( const std::string& url );

    RibbonMenuSearch searcher_;

private:
    void changeTab_( int newTab );

    void sortObjectsRecursive_( std::shared_ptr<Object> object );

    // part of top panel
    void drawHeaderPannel_();

    ImVec2 activeListPos_{ 0,0 };
    bool activeListPressed_{ false };

    void beginTopPanel_();
    void endTopPanel_();
    void drawTopPanelOpened_( bool drawTabs, bool centerItems );

    std::string transformClipboardText_;

    int currentTopPanelHeight_ = 113;
    int topPanelOpenedHeight_ = 113;
    int topPanelHiddenHeight_ = 33;

    ImVec2 sceneSize_{ 310, 0 };
    float informationHeight_{ 0.f };
    float transformHeight_{ 0.f };

    RibbonMenuUIConfig menuUIConfig_;

    // how long blocking window will blink in seconds
    float blockingHighlightTimer_{ 0.0f };

    // current scroll position of tabs panel
    float tabPanelScroll_{ 0.0f };

    bool autoCloseBlockingPlugins_{ true };

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

    std::unordered_map<std::string, CustomContextMenuCheckbox> customCheckBox_;

    std::unique_ptr<Toolbar> toolbar_;
    RibbonNotifier notifier_;
#ifndef __EMSCRIPTEN__
    AsyncRequest asyncRequest_;
#endif // !__EMSCRIPTEN__
};

// Checks if RibbonMenu is available, if it is - forwards notification to RibbonNotifier. Otherwise - calls showModal() function
MRVIEWER_API void pushNotification( const RibbonNotification& notification );

}
