#ifndef MESHLIB_NO_MCP

#include "MRMcp/MRMcp.h"
#include "MRMesh/MROnInit.h"
#include "MRMesh/MRObject.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRRibbonMenuItem.h"
#include "MRViewer/MRRibbonSchema.h"
#include "MRViewer/MRSceneCache.h"
#include "MRViewer/MRStatePlugin.h"

#include <algorithm>
#include <string>
#include <vector>

namespace MR::Mcp
{

// Public type label for a ribbon item. State plugins split by `blocking()` (default: true);
// plain RibbonMenuItem subclasses are one-shot buttons.
static const char* mcpToolType( const RibbonMenuItem& item )
{
    if ( const auto* sp = dynamic_cast<const StateBasePlugin*>( &item ) )
        return sp->blocking() ? "stateBlocking" : "stateNonBlocking";
    return "button";
}

// Mirrors RibbonMenu::getRequirements_: active state plugins are always available (so they can
// be toggled off); otherwise availability comes from `isAvailable(selectedObjects)`.
static std::string mcpToolStatus( const RibbonMenuItem& item,
    const std::vector<std::shared_ptr<const Object>>& selected )
{
    if ( item.isActive() )
        return "available";
    auto reason = item.isAvailable( selected );
    return reason.empty() ? std::string( "available" ) : "disabled: " + reason;
}

static nlohmann::json mcpToolInfoJson( const std::string& id, const MenuItemInfo& info,
    const std::vector<std::shared_ptr<const Object>>& selected )
{
    const auto& item = *info.item;
    std::string tab;
    const int tabIdx = RibbonSchemaHolder::findItemTab( info.item );
    if ( tabIdx >= 0 )
        tab = RibbonSchemaHolder::schema().tabsOrder[tabIdx].name;

    // RibbonMenuItem::getDynamicTooltip overrides the static schema tooltip when non-empty.
    std::string tooltip = item.getDynamicTooltip();
    if ( tooltip.empty() )
        tooltip = info.tooltip;

    return nlohmann::json::object( {
        { "id",       id },
        { "caption",  info.getCaption() },
        { "tab",      std::move( tab ) },
        { "type",     mcpToolType( item ) },
        { "active",   item.isActive() },
        { "status",   mcpToolStatus( item, selected ) },
        { "tooltip",  std::move( tooltip ) },
        { "helpLink", info.helpLink },
    } );
}

static nlohmann::json mcpToolsListAll( const nlohmann::json& )
{
    std::vector<std::string> ids;
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        const auto& items = RibbonSchemaHolder::schema().items;
        ids.reserve( items.size() );
        for ( const auto& [id, info] : items )
            ids.push_back( id );
    } );
    std::sort( ids.begin(), ids.end() );
    return nlohmann::json::object( { { "result", std::move( ids ) } } );
}

static nlohmann::json mcpToolsListActive( const nlohmann::json& )
{
    std::vector<std::string> ids;
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        for ( const auto& [id, info] : RibbonSchemaHolder::schema().items )
        {
            if ( info.item && info.item->isActive() )
                ids.push_back( id );
        }
    } );
    std::sort( ids.begin(), ids.end() );
    return nlohmann::json::object( { { "result", std::move( ids ) } } );
}

static nlohmann::json mcpToolsGetInfo( const nlohmann::json& args )
{
    const auto ids = args.at( "ids" ).get<std::vector<std::string>>();

    nlohmann::json items   = nlohmann::json::array();
    nlohmann::json missing = nlohmann::json::array();
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        const auto& schemaItems = RibbonSchemaHolder::schema().items;
        // Snapshot once — same selection set every iteration.
        const auto& selected = SceneCache::getAllObjects<const Object, ObjectSelectivityType::Selected>();
        for ( const auto& id : ids )
        {
            auto it = schemaItems.find( id );
            if ( it == schemaItems.end() || !it->second.item )
            {
                missing.push_back( id );
                continue;
            }
            items.push_back( mcpToolInfoJson( id, it->second, selected ) );
        }
    } );

    return nlohmann::json::object( { { "result",
        nlohmann::json::object( {
            { "items",   std::move( items ) },
            { "missing", std::move( missing ) },
        } )
    } } );
}

static nlohmann::json mcpToolsAction( const nlohmann::json& args )
{
    const auto id = args.at( "id" ).get<std::string>();
    bool nowActive = false;
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto& items = RibbonSchemaHolder::schema().items;
        auto it = items.find( id );
        if ( it == items.end() || !it->second.item )
            throw std::runtime_error( fmt::format( "tool '{}' not found", id ) );
        auto& item = it->second.item;
        if ( !item->isActive() )
        {
            const auto& selected = SceneCache::getAllObjects<const Object, ObjectSelectivityType::Selected>();
            auto reason = item->isAvailable( selected );
            if ( !reason.empty() )
                throw std::runtime_error( fmt::format( "tools.action {}: {}", id, reason ) );
        }
        item->action();
        nowActive = item->isActive();
    } );
    skipFramesAfterInput();
    return nlohmann::json::object( { { "result",
        nlohmann::json::object( { { "active", nowActive } } )
    } } );
}

MR_ON_INIT{
    Server& server = getDefaultServer();

    server.addTool(
        /*id*/"tools.listAll",
        /*name*/"List all tool ids",
        /*desc*/"Sorted array of every registered tool/plugin id, regardless of which ribbon tab is currently rendered. "
                "Pass interesting ids to `tools.getInfo` for caption / tab / type / status / tooltip, or to "
                "`tools.action` to open or fire the tool. Same id space as the ribbon `ui.pressButton` entries.",
        /*input_schema*/Schema::Object{},
        /*output_schema*/Schema::Array( Schema::String{} ),
        /*func*/mcpToolsListAll
    );

    server.addTool(
        /*id*/"tools.listActive",
        /*name*/"List active tool ids",
        /*desc*/"Sorted array of currently-active tool ids — state plugins whose dialog is open. Empty when no tool is "
                "open. Subset of `tools.listAll`.",
        /*input_schema*/Schema::Object{},
        /*output_schema*/Schema::Array( Schema::String{} ),
        /*func*/mcpToolsListActive
    );

    server.addTool(
        /*id*/"tools.getInfo",
        /*name*/"Get tool metadata (batch)",
        /*desc*/"Look up metadata for one or more tool ids (from `tools.listAll`). "
                "`type`: `\"button\"` (one-shot) | `\"stateBlocking\"` (modal-style dialog) | `\"stateNonBlocking\"`. "
                "`status`: `\"available\"` | `\"disabled: <reason>\"` (active state plugins are always reported "
                "available so they can be toggled off). `tab` is the ribbon tab name or `\"\"` for non-tabbed items "
                "(scene buttons, quick access). `tooltip` prefers a plugin's dynamic tooltip over the static schema "
                "string. Ids absent from the schema land in `missing` instead of erroring the whole call.",
        /*input_schema*/Schema::Object{}.addMember( "ids", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Object{}
            .addMember( "items", Schema::Array(
                Schema::Object{}
                    .addMember( "id",       Schema::String{} )
                    .addMember( "caption",  Schema::String{} )
                    .addMember( "tab",      Schema::String{} )
                    .addMember( "type",     Schema::String{} )
                    .addMember( "active",   Schema::Bool{} )
                    .addMember( "status",   Schema::String{} )
                    .addMember( "tooltip",  Schema::String{} )
                    .addMember( "helpLink", Schema::String{} )
            ) )
            .addMember( "missing", Schema::Array( Schema::String{} ) ),
        /*func*/mcpToolsGetInfo
    );

    server.addTool(
        /*id*/"tools.action",
        /*name*/"Toggle/fire tool by id",
        /*desc*/"Equivalent to clicking the tool's ribbon button. State plugins toggle open/closed; one-shot buttons "
                "run their action. Errors if the id isn't in `tools.listAll`, or if the tool is currently disabled "
                "(message has the same shape as `ui.pressButton` disabled errors). Returns `{active}` — the tool's "
                "post-call active state (always `false` for one-shot buttons).",
        /*input_schema*/Schema::Object{}.addMember( "id", Schema::String{} ),
        /*output_schema*/Schema::Object{}.addMember( "active", Schema::Bool{} ),
        /*func*/mcpToolsAction
    );
}; // MR_ON_INIT

} // namespace MR::Mcp

#endif
