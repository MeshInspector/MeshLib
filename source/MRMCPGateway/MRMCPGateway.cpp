// fastmcpp must be included before any standard headers per MRMcp's pattern.

#if defined( __GNUC__ )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined( _MSC_VER )
#pragma warning( push )
#pragma warning( disable: 4100 )
#pragma warning( disable: 4355 )
#endif

#include <fastmcpp.hpp>
#include <fastmcpp/proxy.hpp>
#include <fastmcpp/server/stdio_server.hpp>
#include <fastmcpp/client/transports.hpp>
#include <fastmcpp/mcp/handler.hpp>

#if defined( __GNUC__ )
#pragma GCC diagnostic pop
#elif defined( _MSC_VER )
#pragma warning( pop )
#endif

#include "MRMCPGatewaySpawn.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace MR::McpGateway
{

namespace
{

struct Config
{
    std::string targetUrl    = "http://127.0.0.1:7887";
    std::string ssePath      = "/sse";
    std::string messagesPath = "/messages";
    std::filesystem::path launchCommand;
    std::vector<std::string> launchArgs;
    std::chrono::seconds launchTimeout{ 30 };
};

bool probeBackendAlive( const std::string& targetUrl )
{
    httplib::Client cli( targetUrl );
    cli.set_connection_timeout( std::chrono::milliseconds( 500 ) );
    cli.set_read_timeout( std::chrono::milliseconds( 500 ) );
    auto res = cli.Get( "/__mrmcpgateway_probe" );
    return static_cast<bool>( res );
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

// Single source of truth for "is the backend currently alive?". Updated by every probe.
// On a transition (alive ↔ dead), we synchronously emit `notifications/tools/list_changed`
// so the proxied tool set surfaces / disappears in the connected MCP client without polling.
std::atomic<bool> g_backendAlive{ false };
// Suppresses the very first probe's transition emit, so we don't spuriously fire
// `list_changed` during the client's initial `tools/list` (whose response already carries
// the right tool set anyway).
std::atomic<bool> g_backendPrimed{ false };

// Probe the backend, update g_backendAlive, and on a state transition synchronously emit
// `tools/list_changed`. Use this in place of probeBackendAlive() at every call site.
bool probeAndTrackBackend( const std::string& targetUrl )
{
    const bool nowAlive = probeBackendAlive( targetUrl );
    const bool wasAlive = g_backendAlive.exchange( nowAlive );
    const bool wasPrimed = g_backendPrimed.exchange( true );
    if ( wasPrimed && wasAlive != nowAlive )
        emitToolsListChanged();
    return nowAlive;
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
        if ( req.is_object() && req.value( "method", std::string{} ) == "initialize" )
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
        return inner( req );
    };
    fastmcpp::server::StdioServerWrapper server( handler );
    return server.run() ? 0 : 1;
}
