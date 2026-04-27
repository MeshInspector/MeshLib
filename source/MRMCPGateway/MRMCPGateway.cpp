// Must not include any standard headers before MRFastmcpp.h (fastmcpp's macro
// shenanigans rely on it).
#include "MRPch/MRFastmcpp.h"

#include "MRMCPGatewaySpawn.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <pwd.h>
#include <unistd.h>
#endif

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
    std::string toolsCacheNamespace; // --tools-cache-namespace <name> (optional sub-folder)
};

// Compile-time build stamp baked into this translation unit. Written to a
// sibling .stamp file after each successful cache prime; compared on startup
// to decide whether the existing cache is still good. To force a re-prime,
// touch this file (or just delete the cache dir).
constexpr const char* kBuildStamp = __TIMESTAMP__;

// PASTED from MR::getUserConfigDir() in MeshLib/source/MRMesh/MRSystem.cpp.
// Adapted to: hardcode the leaf folder (no Config::instance() dep) and use
// std::cerr instead of spdlog so the gateway keeps its zero-MRMesh,
// zero-spdlog dependency footprint.
std::filesystem::path gatewayUserConfigDir()
{
#if defined( _WIN32 )
    std::filesystem::path filepath( _wgetenv( L"APPDATA" ) );
#else
#if defined( __EMSCRIPTEN__ )
    std::filesystem::path filepath( "/" );
#else
    std::filesystem::path filepath;
    if ( const auto* pw = getpwuid( getuid() ) )
        filepath = pw->pw_dir;
    else if ( const char* h = std::getenv( "HOME" ) )
        filepath = h;
#endif
    filepath /= ".local";
    filepath /= "share";
#endif
    filepath /= "MRMCPGateway";
    std::error_code ec;
    if ( !std::filesystem::is_directory( filepath, ec ) || ec )
    {
        std::filesystem::create_directories( filepath, ec );
        if ( ec )
            std::cerr << "MRMCPGateway: cannot create " << filepath.string()
                      << ": " << ec.message() << "\n";
    }
    return filepath;
}

std::filesystem::path cacheDir( const Config& cfg )
{
    auto d = gatewayUserConfigDir();
    if ( !cfg.toolsCacheNamespace.empty() )
        d /= cfg.toolsCacheNamespace;
    return d;
}
std::filesystem::path cachePath( const Config& cfg ) { return cacheDir( cfg ) / "mcp_tools_cache.json"; }

// Cached tool entries loaded from the cache file at startup; spliced into
// `tools/list` responses when the backend is offline.
std::vector<nlohmann::json> g_cachedTools;

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

// Reads the cache file and returns true iff its `stamp` field equals kBuildStamp.
bool cacheStampMatches( const std::filesystem::path& cache )
{
    std::ifstream f( cache );
    if ( !f )
        return false;
    nlohmann::json doc;
    try { f >> doc; } catch ( ... ) { return false; }
    if ( !doc.is_object() || !doc.contains( "stamp" ) || !doc["stamp"].is_string() )
        return false;
    return doc["stamp"].get<std::string>() == kBuildStamp;
}

// Reads the cache, sets its `stamp` field to kBuildStamp, atomically writes it back.
bool embedStampInCache( const std::filesystem::path& cache )
{
    std::ifstream in( cache );
    if ( !in )
        return false;
    nlohmann::json doc;
    try { in >> doc; } catch ( ... ) { return false; }
    in.close();
    if ( !doc.is_object() )
        return false;
    doc["stamp"] = kBuildStamp;

    auto tmp = cache;
    tmp += ".tmp";
    {
        std::ofstream out( tmp );
        if ( !out )
            return false;
        out << doc.dump( 2 );
    }
    std::error_code ec;
    std::filesystem::rename( tmp, cache, ec );
    if ( ec )
    {
        std::filesystem::remove( tmp );
        return false;
    }
    return true;
}

// If the cache is stale or missing AND the backend is offline, spawn the backend
// synchronously (attached, hidden) so it dumps the tool schemas, then embed our
// build stamp into the just-written file. Best-effort — failures are logged but
// non-fatal (the gateway proceeds with whatever cache content exists, or none).
void ensureFreshCache( const Config& cfg )
{
    const auto cache = cachePath( cfg );

    if ( cacheStampMatches( cache ) )
        return; // already fresh

    // Backend already running → live tools/list will be authoritative this
    // session, skip the prime (and avoid port collision).
    if ( probeBackendAlive( cfg.targetUrl ) )
        return;

    if ( cfg.launchCommand.empty() )
        return;

    std::error_code ec;
    const auto mtimeBefore = std::filesystem::exists( cache, ec )
        ? std::filesystem::last_write_time( cache, ec )
        : std::filesystem::file_time_type{};

    std::vector<std::string> primeArgs = cfg.launchArgs;
    for ( const char* flag : { "-hidden", "-noEventLoop", "-noTelemetry", "-noSplash" } )
        primeArgs.emplace_back( flag );
    primeArgs.emplace_back( "-mcpDumpFile" );
    primeArgs.emplace_back( cache.string() );

    // Synchronous spawn: blocks until the backend exits (or times out). Avoids
    // the polling-sees-stale-cache race that file-existence polling has when an
    // older cache file is already on disk.
    if ( !spawnAndWait( cfg.launchCommand, primeArgs, cfg.launchTimeout ) )
    {
        std::cerr << "MRMCPGateway: prime spawn did not complete cleanly\n";
        return;
    }

    // Confirm the backend actually rewrote the cache file before stamping it.
    if ( !std::filesystem::exists( cache, ec ) )
    {
        std::cerr << "MRMCPGateway: backend exited without writing cache file\n";
        return;
    }
    const auto mtimeAfter = std::filesystem::last_write_time( cache, ec );
    if ( mtimeAfter <= mtimeBefore )
    {
        std::cerr << "MRMCPGateway: cache file unchanged after prime; not stamping\n";
        return;
    }
    embedStampInCache( cache );
}

void loadCachedTools( const Config& cfg )
{
    g_cachedTools.clear();
    const auto cache = cachePath( cfg );
    std::ifstream f( cache );
    if ( !f )
        return;
    nlohmann::json doc;
    try { f >> doc; } catch ( ... ) { return; }
    if ( !doc.is_object() || !doc.contains( "tools" ) || !doc["tools"].is_array() )
        return;
    for ( const auto& t : doc["tools"] )
        if ( t.is_object() )
            g_cachedTools.push_back( t );
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
