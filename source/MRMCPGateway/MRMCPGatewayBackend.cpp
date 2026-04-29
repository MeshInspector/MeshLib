#include "MRMCPGatewayBackend.h"

#include "MRMCPGatewayConfig.h"
#include "MRMCPGatewaySpawn.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <thread>

namespace MR::McpGateway
{

std::atomic<bool>& getBackendAlive()
{
    static std::atomic<bool> instance{ false };
    return instance;
}

namespace
{

// Suppresses the very first probe's transition emit, so we don't spuriously fire
// `list_changed` during the client's initial `tools/list` (whose response already
// carries the right tool set anyway).
std::atomic<bool>& backendPrimed()
{
    static std::atomic<bool> instance{ false };
    return instance;
}

void emitToolsListChanged()
{
    const nlohmann::json notif = {
        { "jsonrpc", "2.0" },
        { "method",  "notifications/tools/list_changed" },
        { "params",  nlohmann::json::object() },
    };
    std::cout << notif.dump() << std::endl;
    std::cout.flush();
}

nlohmann::json emptyObjectSchema()
{
    return {
        { "type", "object" },
        { "properties", nlohmann::json::object() },
        { "required", nlohmann::json::array() },
    };
}

nlohmann::json stringSchema()
{
    return { { "type", "string" } };
}

nlohmann::json launchInputSchema()
{
    return {
        { "type", "object" },
        { "properties", {
            { "args", {
                { "type", "array" },
                { "items", { { "type", "string" } } },
                { "description", "Override the args forwarded to the backend for this call. "
                                 "Defaults to the gateway's --launch-arg values." },
            } },
        } },
        { "required", nlohmann::json::array() },
    };
}

} // anonymous namespace

bool probeBackendAlive( const std::string& targetUrl )
{
    httplib::Client cli( targetUrl );
    cli.set_connection_timeout( std::chrono::milliseconds( 500 ) );
    cli.set_read_timeout( std::chrono::milliseconds( 500 ) );
    auto res = cli.Get( "/__mrmcpgateway_probe" );
    return static_cast<bool>( res );
}

bool probeAndTrackBackend( const std::string& targetUrl )
{
    const bool nowAlive = probeBackendAlive( targetUrl );
    const bool wasAlive = getBackendAlive().exchange( nowAlive );
    const bool wasPrimed = backendPrimed().exchange( true );
    if ( wasPrimed && wasAlive != nowAlive )
        emitToolsListChanged();
    return nowAlive;
}

void registerLocalTools( fastmcpp::ProxyApp& proxy, const Config& cfg )
{
    proxy.local_tools().register_tool( fastmcpp::tools::Tool(
        std::string( "launch" ),
        launchInputSchema(),
        stringSchema(),
        [cfg]( const fastmcpp::Json& input ) -> fastmcpp::Json
        {
            std::vector<std::string> args = cfg.launchArgs;
            if ( input.is_object() )
            {
                if ( auto it = input.find( "args" ); it != input.end() && it->is_array() )
                {
                    args.clear();
                    for ( const auto& a : *it )
                        if ( a.is_string() )
                            args.push_back( a.get<std::string>() );
                }
            }
            // Always tell MI the port the gateway will probe; last-occurrence-wins
            // parsing on MI's side means this beats any user-supplied -mcpPort in args.
            args.emplace_back( "-mcpPort" );
            args.emplace_back( std::to_string( cfg.mcpPort ) );
            if ( probeAndTrackBackend( cfg.targetUrl ) )
                return std::string( "already running" );
            if ( !spawnDetached( cfg.launchCommand, args ) )
                return std::string( "failed: spawn error" );
            const auto deadline = std::chrono::steady_clock::now() + cfg.launchTimeout;
            while ( std::chrono::steady_clock::now() < deadline )
            {
                if ( probeAndTrackBackend( cfg.targetUrl ) )
                    return std::string( "started" );
                std::this_thread::sleep_for( std::chrono::milliseconds( 250 ) );
            }
            return std::string( "timeout: backend not ready within " )
                 + std::to_string( cfg.launchTimeout.count() ) + "s";
        },
        std::optional<std::string>( "Launch backend" ),
        std::optional<std::string>( "Start the MCP backend application that this gateway proxies to. "
                                    "Optionally pass 'args' to override the gateway's --launch-arg defaults for this call (the executable path itself is fixed by the gateway operator). "
                                    "Returns 'started', 'already running', 'failed: <reason>', or 'timeout: <reason>'." ),
        std::nullopt,
        std::vector<std::string>{}
    ) );

    proxy.local_tools().register_tool( fastmcpp::tools::Tool(
        std::string( "status" ),
        emptyObjectSchema(),
        stringSchema(),
        [cfg]( const fastmcpp::Json& ) -> fastmcpp::Json
        {
            return std::string( probeAndTrackBackend( cfg.targetUrl ) ? "running" : "not started" );
        },
        std::optional<std::string>( "Backend status" ),
        std::optional<std::string>( "Returns 'running' if the backend MCP server is reachable, 'not started' otherwise." ),
        std::nullopt,
        std::vector<std::string>{}
    ) );
}

} // namespace MR::McpGateway
