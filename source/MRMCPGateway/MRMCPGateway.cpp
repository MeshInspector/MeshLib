// Must not include any standard headers before MRFastmcpp.h (fastmcpp's macro
// shenanigans rely on it).
#include "MRMcp/MRFastmcpp.h"

#include "MRMCPGatewayBackend.h"
#include "MRMCPGatewayCache.h"
#include "MRMCPGatewayConfig.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

namespace MR::McpGateway
{

namespace
{

void printUsage()
{
    std::cerr <<
        "Usage: MRMCPGateway --launch-cmd <path> [options]\n"
        "  --launch-cmd <path>      Required. Backend executable launched by the 'launch' tool.\n"
        "                           Fixed at startup; not overridable via tool call.\n"
        "  --launch-arg <value>     Default argument forwarded to the backend (repeatable).\n"
        "                           A 'launch' tool call may override these for that call.\n"
        "  --launch-timeout <secs>  How long 'launch' waits for the backend (default 30).\n"
        "  --target-url <url>       Backend MCP server URL (default http://127.0.0.1:7887).\n"
        "  --sse-path <path>        SSE endpoint path (default /sse).\n"
        "  --messages-path <path>   POST endpoint path (default /messages).\n"
        "  --tools-cache-namespace <name>\n"
        "                           Optional sub-folder under the gateway's user-data dir,\n"
        "                           letting multiple installations keep independent caches.\n"
        "  --help, -h               Show this message.\n";
}

bool parseArgs( int argc, char** argv, Config& cfg )
{
    for ( int i = 1; i < argc; ++i )
    {
        const std::string a = argv[i];
        const auto needNext = [&]( const char* what ) -> bool
        {
            if ( i + 1 >= argc )
            {
                std::cerr << "MRMCPGateway: " << what << " requires a value\n";
                return false;
            }
            return true;
        };

        if ( a == "--target-url" )
        {
            if ( !needNext( "--target-url" ) ) return false;
            cfg.targetUrl = argv[++i];
        }
        else if ( a == "--sse-path" )
        {
            if ( !needNext( "--sse-path" ) ) return false;
            cfg.ssePath = argv[++i];
        }
        else if ( a == "--messages-path" )
        {
            if ( !needNext( "--messages-path" ) ) return false;
            cfg.messagesPath = argv[++i];
        }
        else if ( a == "--launch-cmd" )
        {
            if ( !needNext( "--launch-cmd" ) ) return false;
            cfg.launchCommand = argv[++i];
        }
        else if ( a == "--launch-arg" )
        {
            if ( !needNext( "--launch-arg" ) ) return false;
            cfg.launchArgs.emplace_back( argv[++i] );
        }
        else if ( a == "--launch-timeout" )
        {
            if ( !needNext( "--launch-timeout" ) ) return false;
            cfg.launchTimeout = std::chrono::seconds( std::atoi( argv[++i] ) );
        }
        else if ( a == "--tools-cache-namespace" )
        {
            if ( !needNext( "--tools-cache-namespace" ) ) return false;
            cfg.toolsCacheNamespace = argv[++i];
        }
        else if ( a == "--help" || a == "-h" )
        {
            printUsage();
            std::exit( 0 );
        }
        else
        {
            std::cerr << "MRMCPGateway: unknown argument: " << a << "\n";
            printUsage();
            return false;
        }
    }

    if ( cfg.launchCommand.empty() )
    {
        std::cerr << "MRMCPGateway: --launch-cmd is required\n";
        printUsage();
        return false;
    }
    return true;
}

} // anonymous namespace

} // namespace MR::McpGateway

int main( int argc, char** argv )
{
    using namespace MR::McpGateway;

    Config cfg;
    if ( !parseArgs( argc, argv, cfg ) )
        return 1;

    // Prime the on-disk tool cache (synchronous; ~3-5 s when actually priming) and
    // load the resulting JSON into memory. Failures are non-fatal: we proceed with
    // an empty cache and only the local `launch`/`status` tools will be visible
    // until the backend actually launches.
    ensureFreshCache( cfg );
    loadCachedTools( cfg );

    fastmcpp::ProxyApp proxy(
        [cfg]() {
            // Fast pre-check: SseClientTransport blocks for a long internal timeout when the
            // backend is offline. Throw quickly so ProxyApp's per-method catch block falls
            // back to local-only listings instead of hanging the stdio loop. The probe also
            // updates g_backendAlive and fires `tools/list_changed` on transitions, so a forwarded
            // tool call that discovers the backend has died emits the disappearance synchronously.
            if ( !probeAndTrackBackend( cfg.targetUrl ) )
                throw std::runtime_error( "backend unreachable at " + cfg.targetUrl );
            auto transport = std::make_unique<fastmcpp::client::SseClientTransport>(
                cfg.targetUrl, cfg.ssePath, cfg.messagesPath );
            return fastmcpp::client::Client( std::move( transport ) );
        },
        std::string( "MRMCPGateway" ),
        std::string( "0.1" )
    );

    registerLocalTools( proxy, cfg );

    auto inner = fastmcpp::mcp::make_mcp_handler( proxy );
    // We hand-craft the `initialize` response for two reasons:
    //  1. Advertise `tools.listChanged: true` so MCP clients honour our list-changed
    //     notifications (fastmcpp's default initialize sets `"tools": {}` empty).
    //  2. Avoid fastmcpp's initialize handler calling `proxy.list_all_resources/templates/prompts`,
    //     which each invoke our client factory and probe the backend — making `initialize` slow
    //     enough to trip `claude mcp list`'s health-check timeout when the backend is offline.
    auto handler = [inner]( const fastmcpp::Json& req ) -> fastmcpp::Json
    {
        const std::string method = req.is_object() ? req.value( "method", std::string{} ) : std::string{};
        if ( method == "initialize" )
        {
            const auto id = req.contains( "id" ) ? req.at( "id" ) : fastmcpp::Json();
            return fastmcpp::Json{
                { "jsonrpc", "2.0" },
                { "id", id },
                { "result", {
                    { "protocolVersion", "2024-11-05" },
                    { "capabilities", { { "tools", { { "listChanged", true } } } } },
                    { "serverInfo", { { "name", "MRMCPGateway" }, { "version", "0.1" } } },
                } },
            };
        }

        fastmcpp::Json resp = inner( req );

        // When the backend is offline, fastmcpp's proxy returns only our local tools
        // (`launch`, `status`). Splice in the cached schema array so the MCP client
        // still sees the full proxied surface and can decide which tools to call.
        if ( method == "tools/list" && !g_backendAlive.load() && !g_cachedTools.empty()
             && resp.is_object() && resp.contains( "result" ) && resp["result"].contains( "tools" )
             && resp["result"]["tools"].is_array() )
        {
            auto& tools = resp["result"]["tools"];
            std::set<std::string> seen;
            for ( const auto& t : tools )
                if ( t.is_object() && t.contains( "name" ) && t["name"].is_string() )
                    seen.insert( t["name"].get<std::string>() );
            for ( const auto& cached : g_cachedTools )
                if ( cached.contains( "name" ) && cached["name"].is_string()
                     && !seen.count( cached["name"].get<std::string>() ) )
                    tools.push_back( cached );
        }
        return resp;
    };
    fastmcpp::server::StdioServerWrapper server( handler );
    return server.run() ? 0 : 1;
}
