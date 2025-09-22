#pragma once
#include "MRViewerFwd.h"
#include "MRRibbonRegisterItem.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRphmap.h"
#include <json/forwards.h>
#include <filesystem>
#include <vector>
#include <string>
#include <memory>

namespace MR
{

// needed for big buttons text aligning
using SplitCaptionInfo = std::vector<std::pair<std::string_view, float>>;

struct MenuItemCaptionSize
{
    float baseSize{ 0.0f };
    SplitCaptionInfo splitInfo;
};

struct MenuItemInfo
{
    std::shared_ptr<RibbonMenuItem> item;
    std::string caption;
    std::string tooltip;
    std::string icon;
    MenuItemCaptionSize captionSize; // already scaled
    std::string helpLink; // link to help page
};

/// interface for plugins that should be notified when their information is loaded from the schema json file
/// allows to alternate behavior depending on whether the plugin is available in the app
class MRVIEWER_CLASS RibbonSchemaLoadListener
{
public:
    virtual ~RibbonSchemaLoadListener() = default;

protected:
    friend class RibbonSchemaLoader;
    virtual void onRibbonSchemaLoad_() = 0;
};

using ItemMap = HashMap<std::string, MenuItemInfo>;
using TabsGroupsMap = HashMap<std::string, std::vector<std::string>>;
using GroupsItemsMap = TabsGroupsMap;
using MenuItemsList = std::vector<std::string>;
struct RibbonTab
{
    std::string name;
    int priority{ 0 };
    bool experimental{ false };
};

// This structure describes UI schema of ribbon menu
struct RibbonSchema
{
    std::vector<RibbonTab> tabsOrder;
    TabsGroupsMap tabsMap;
    GroupsItemsMap groupsMap;
    ItemMap items;
    MenuItemsList defaultQuickAccessList;
    MenuItemsList headerQuickAccessList;
    MenuItemsList sceneButtonsList;

    /// deletes empty groups (and references on them from tabs)
    MRVIEWER_API void eliminateEmptyGroups();

    /// re-order items in tabsOrder according to their priority, the order of items having same priority is preserved
    MRVIEWER_API void sortTabsByPriority();

    /// updates inner caption for all StateBasePlugins in schema
    MRVIEWER_API void updateCaptions();
};

// This class holds static ribbon schema,
// note that schema becomes valid after loading (RibbonSchemaLoader)
class MRVIEWER_CLASS RibbonSchemaHolder
{
public:
    MRVIEWER_API static RibbonSchema& schema();

    /// adds item to static holder (needed to be independent of construction time)
    /// returns false if item with such name is already present
    MRVIEWER_API static bool addItem( const std::shared_ptr<RibbonMenuItem>& item );

    /// removes item from the static holder
    /// returns false if item was not present
    MRVIEWER_API static bool delItem( const std::shared_ptr<RibbonMenuItem>& item );

    /// struct to hold information for search result presentation
    struct SearchResult
    {
        int tabIndex{ -1 }; // -1 is default value if item has no tab
        const MenuItemInfo* item{ nullptr }; // item info to show correct caption
    };

    /// ancillary struct to hold information for search result order
    struct SearchResultWeight
    {
        float captionWeight{ 1.f };
        float captionOrderWeight{ 1.f };
        float tooltipWeight{ 1.f };
        float tooltipOrderWeight{ 1.f };
    };

    /// tool search options
    struct SearchParams
    {
        /// gets the number of matches in the captions
        int* captionCount = nullptr;
        /// returns the coefficients of matching the search string for each tool found
        std::vector<SearchResultWeight>* weights = nullptr;
        /// sort the results according to the launch capability (empty requirements)
        RequirementsFunction requirementsFunc = {};
    };
    /// searches for tools with the most appropriate names (captions) or tooltips
    MRVIEWER_API static std::vector<SearchResult> search( const std::string& searchStr, const SearchParams& params );

    /// returns item tab index in schema.tabsOrder or -1 if no tab found (e.g. scene fast access panel or header access panel)
    MRVIEWER_API static int findItemTab( const std::shared_ptr<RibbonMenuItem>& item );

private:
    RibbonSchemaHolder() = default;
};

// Class for loading ribbon schema from structure files (basically called from RibbonMenu, but can be called separately)
class MRVIEWER_CLASS RibbonSchemaLoader
{
public:
    RibbonSchemaLoader() = default;
    virtual ~RibbonSchemaLoader() = default;

    // Loads schema from files
    MRVIEWER_API virtual void loadSchema() const;

    // Reads schema items list from `root` to `list`
    MRVIEWER_API static void readMenuItemsList( const Json::Value& root, MenuItemsList& list );

    // Recalc items sizes
    MRVIEWER_API static void recalcItemSizes();
protected:
    // finds structure json files in exe directory
    MRVIEWER_API virtual std::vector<std::filesystem::path> getStructureFiles_( const std::string& fileExtension ) const;
    // sort structure json files by order
    MRVIEWER_API void sortFilesByOrder_( std::vector<std::filesystem::path>& files ) const;
    // appends one menu items json info
    MRVIEWER_API void readItemsJson_( const std::filesystem::path& path ) const;
    MRVIEWER_API void readItemsJson_( const Json::Value& root ) const;
    // appends one ui json info
    MRVIEWER_API void readUIJson_( const std::filesystem::path& path ) const;
    MRVIEWER_API void readUIJson_( const Json::Value& root ) const;
};


template<typename T>
class RibbonMenuItemCall
{
    static_assert( std::is_base_of_v<RibbonMenuItem, T> );
public:
    template<typename F>
    RibbonMenuItemCall( F f, std::function<void(std::shared_ptr<T>)> g ) : g_( std::move( g ) )
    {
        const auto& items = RibbonSchemaHolder::schema().items;
        for ( const auto& item : items )
        {
            auto plugin = std::dynamic_pointer_cast< T >( item.second.item );
            if ( !plugin )
                continue;
            f( plugin_ = std::move( plugin ) );
            break;
        }
    }
    ~RibbonMenuItemCall()
    {
        if ( g_ && plugin_ )
            g_( plugin_ );
    }
private:
    std::shared_ptr<T> plugin_;
    std::function<void(std::shared_ptr<T>)> g_;
};

/// calls f(const std::shared_ptr<plugin> &) on module loading, and calls g(const std::shared_ptr<plugin> &) on module unloading
#define MR_RIBBON_ITEM_CALL(pluginType,f,g) \
    static MR::RibbonMenuItemCall<pluginType> ribbonMenuItemCall##func##pluginType##_( f, g );

} //namespace MR
