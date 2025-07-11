#include "MRRibbonLayoutConfig.h"
#include "MRColorTheme.h"
#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "MRRibbonMenu.h"
#include "MRMesh/MRSerializer.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

RibbonConfig createRibbonConfigFromJson( const Json::Value& root )
{
    RibbonConfig config;
    if ( root["MenuLayout"].isObject() )
    {
        const auto& layout = root["MenuLayout"];
        config.menuUIConfig = RibbonMenuUIConfig();
        auto& menuConfig = *config.menuUIConfig;
        if ( layout["drawTabs"].isBool() )
            menuConfig.topLayout = layout["drawTabs"].asBool() ? RibbonTopPanelLayoutMode::RibbonWithTabs : RibbonTopPanelLayoutMode::RibbonNoTabs;
        if ( layout["centerRibbonItems"].isBool() )
            menuConfig.centerRibbonItems = layout["centerRibbonItems"].asBool();
        if ( layout["drawToolbar"].isBool() )
            menuConfig.drawToolbar = layout["drawToolbar"].asBool();
        if ( layout["drawLeftPanel"].isBool() )
            menuConfig.drawScenePanel = layout["drawLeftPanel"].asBool();
        if ( layout["drawViewportTags"].isBool() )
            menuConfig.drawViewportTags = layout["drawViewportTags"].asBool();
        if ( layout["drawNotifications"].isBool() )
            menuConfig.drawNotifications = layout["drawNotifications"].asBool();
    }

    Color monochrome( 0, 0, 0, 0 );
    deserializeFromJson( root["monochromeRibbonIcons"], monochrome );
    if ( monochrome.a != 0 )
        config.monochromeRibbonIcons = monochrome;

    if ( root["ColorTheme"].isObject() )
        config.colorTheme = root["ColorTheme"];

    if ( root["RibbonStructure"].isObject() )
        config.ribbonStructure = root["RibbonStructure"];

    if ( root["RibbonItemsOverrides"].isObject() )
        config.ribbonItemsOverrides = root["RibbonItemsOverrides"];

    return config;
}

void applyRibbonConfig( const RibbonConfig& config )
{
    auto& viewer = getViewerInstance();
    auto ribbonMenu = RibbonMenu::instance();
    if ( !ribbonMenu )
    {
        spdlog::warn( "Cannot apply layout config" );
        return;
    }

    if ( config.menuUIConfig )
        ribbonMenu->setMenuUIConfig( *config.menuUIConfig );

    if ( config.monochromeRibbonIcons )
        ribbonMenu->getRibbonButtonDrawer().setMonochrome( config.monochromeRibbonIcons );

    if ( config.colorTheme )
    {
        ColorTheme::setupFromJson( *config.colorTheme );
        ColorTheme::apply();
    }

    if ( config.ribbonStructure || config.ribbonItemsOverrides )
    {
        class ConfigLoader : public RibbonSchemaLoader
        {
        public:
            void updateUIStructure( const Json::Value& root )
            {
                RibbonSchemaHolder::schema().tabsMap.clear();
                RibbonSchemaHolder::schema().tabsOrder.clear();
                RibbonSchemaHolder::schema().groupsMap.clear();
                RibbonSchemaHolder::schema().sceneButtonsList.clear();
                RibbonSchemaHolder::schema().headerQuickAccessList.clear();
                RibbonSchemaHolder::schema().defaultQuickAccessList.clear();
                readUIJson_( root );
            }
            void updateItemsStructure( const Json::Value& root )
            {
                const auto& items = root["Items"];
                if ( !items.isArray() )
                    return;
                auto itemsSize = int( items.size() );
                for ( int i = 0; i < itemsSize; ++i )
                {
                    const auto& item = items[i];
                    auto& itemName = item["Name"];
                    auto findIt = RibbonSchemaHolder::schema().items.find( itemName.asString() );
                    if ( findIt == RibbonSchemaHolder::schema().items.end() )
                        continue;
                    findIt->second.caption = "";
                    findIt->second.helpLink = "";
                    findIt->second.item->setDropItemsFromItemList( {} );
                }
                readItemsJson_( root );
            }
        };
        CommandLoop::appendCommand( [uiSchema = config.ribbonStructure, itemsOverrides = config.ribbonItemsOverrides]
        {
            ConfigLoader loader;
            if ( uiSchema )
                loader.updateUIStructure( *uiSchema );

            if ( itemsOverrides )
                loader.updateItemsStructure( *itemsOverrides );

            loader.recalcItemSizes();

            RibbonSchemaHolder::schema().eliminateEmptyGroups();
            RibbonSchemaHolder::schema().sortTabsByPriority();
            RibbonSchemaHolder::schema().updateCaptions();
        } );
    }
    viewer.incrementForceRedrawFrames( viewer.forceRedrawMinimumIncrementAfterEvents, viewer.swapOnLastPostEventsRedraw );
}

}
