#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRPch/MRJson.h"
#include "MRMesh/MRphmap.h"
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
    // if enabled, shows experimental features in ribbon menu, and in search results
    bool experimentalFeatures{ false };
};

// This class holds static ribbon schema,
// note that schema becomes valid after loading (RibbonSchemaLoader)
class MRVIEWER_CLASS RibbonSchemaHolder
{
public:
    MRVIEWER_API static RibbonSchema& schema();

    // adds item to static holder (needed to be independent of construction time)
    // returns false if item with such name is already present
    MRVIEWER_API static bool addItem( std::shared_ptr<RibbonMenuItem> item );

    // struct to hold information for search result presentation
    struct SearchResult
    {
        int tabIndex{ -1 }; // -1 is default value if item has no tab
        const MenuItemInfo* item{ nullptr }; // item info to show correct caption
    };
    // ancillary struct to hold information for search result order
    struct SearchResultWeight
    {
        float captionWeight{ 1.f };
        float captionOrderWeight{ 1.f };
        float tooltipWeight{ 1.f };
        float tooltipOrderWeight{ 1.f };
    };
    // 
    MRVIEWER_API static std::vector<SearchResult> search( const std::string& searchStr, int* captionCount = nullptr,
        std::vector<SearchResultWeight>* weights = nullptr );

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
    // appends one ui json info
    MRVIEWER_API void readUIJson_( const std::filesystem::path& path ) const;
};

}