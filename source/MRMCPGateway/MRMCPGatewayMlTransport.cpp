#include "MRMCPGatewayMlTransport.h"
#include "MRMCPGatewayBackend.h"

#include <chrono>
#include <iostream>

namespace MR::McpGateway
{

namespace
{

constexpr auto kSseConnectTimeout   = std::chrono::seconds( 10 );
constexpr auto kSseReadTimeout      = std::chrono::seconds( 300 );
constexpr auto kPostConnectTimeout  = std::chrono::seconds( 5 );
constexpr auto kPostReadTimeout     = std::chrono::seconds( 30 );
constexpr int  kReconnectIntervalMs = 1000;
// Briefly wait for the SSE listener to capture (or recapture) a session_id
// before failing a request. Covers the gap between gateway startup / `launch`
// completing / backend restarting and the next SSE reconnect arriving.
constexpr auto kSessionWaitTimeout  = std::chrono::seconds( 3 );

} // anonymous namespace

// Minimal "http://host:port" URL parser. Only called on `cfg.targetUrl`, so
// we don't need a full URL grammar.
MLClientTransport::HostPort MLClientTransport::parseHostPort( const std::string& url )
{
    HostPort out;
    std::string rest = url;
    if ( rest.rfind( "http://", 0 ) == 0 )
        rest = rest.substr( 7 );
    else if ( rest.rfind( "https://", 0 ) == 0 )
        rest = rest.substr( 8 );

    auto colon = rest.find( ':' );
    auto slash = rest.find( '/' );
    if ( slash == std::string::npos )
        slash = rest.size();

    if ( colon != std::string::npos && colon < slash )
    {
        out.host = rest.substr( 0, colon );
        try { out.port = std::stoi( rest.substr( colon + 1, slash - colon - 1 ) ); }
        catch ( ... ) { out.port = 80; }
    }
    else
    {
        out.host = rest.substr( 0, slash );
    }
    return out;
}

MLClientTransport::MLClientTransport( const std::string& targetUrl,
                                      const std::string& ssePath,
                                      const std::string& messagesPath )
    : MLClientTransport( parseHostPort( targetUrl ), ssePath, messagesPath )
{}

MLClientTransport::MLClientTransport( const HostPort& hp,
                                      const std::string& ssePath,
                                      const std::string& messagesPath )
    : httpForSse_( hp.host.c_str(), hp.port )
    , sse_( httpForSse_, ssePath )
    , httpForPost_( hp.host.c_str(), hp.port )
    , messagesPath_( messagesPath )
{
    httpForSse_.set_connection_timeout( static_cast<time_t>( kSseConnectTimeout.count() ), 0 );
    httpForSse_.set_read_timeout( static_cast<time_t>( kSseReadTimeout.count() ), 0 );
    httpForSse_.set_keep_alive( true );

    httpForPost_.set_connection_timeout( static_cast<time_t>( kPostConnectTimeout.count() ), 0 );
    httpForPost_.set_read_timeout( static_cast<time_t>( kPostReadTimeout.count() ), 0 );
    httpForPost_.set_keep_alive( true );

    sse_.on_event( "endpoint", [this]( const httplib::sse::SSEMessage& msg )
    {
        // msg.data == "/messages?session_id=<sid>"
        const std::string key = "session_id=";
        auto pos = msg.data.find( key );
        if ( pos == std::string::npos )
            return;
        const auto sidStart = pos + key.size();
        const auto sidEnd = msg.data.find_first_of( "&#", sidStart );
        std::string sid = msg.data.substr( sidStart,
            ( sidEnd == std::string::npos ) ? std::string::npos : ( sidEnd - sidStart ) );
        // Set sessionId_ before flipping alive: when updateBackendAliveAndNotify
        // emits `tools/list_changed`, the client's follow-up `tools/list` will
        // route through the alive path of request(), which needs sessionId_
        // already populated.
        {
            std::lock_guard<std::mutex> lk( sessionMutex_ );
            sessionId_ = std::move( sid );
        }
        // updateBackendAliveAndNotify (not bare store): emits `tools/list_changed`
        // on dead→alive transitions, so externally-launched MI surfaces tools to
        // the MCP client without requiring a `launch` tool call.
        updateBackendAliveAndNotify( true );
        sessionCv_.notify_all();
    } );

    sse_.on_error( [this]( httplib::Error /*err*/ )
    {
        // Wipe sessionId_ so subsequent request() blocks on the cv until the
        // SSEClient's auto-reconnect produces a fresh one — without this, after
        // a backend restart we'd send the old session_id and the server would
        // 404 it.
        {
            std::lock_guard<std::mutex> lk( sessionMutex_ );
            sessionId_.clear();
        }
        // alive→dead transition: emit `tools/list_changed` so the MCP client
        // re-fetches and sees the cached-only tool surface (or fewer tools).
        updateBackendAliveAndNotify( false );
        sessionCv_.notify_all();
    } );

    sse_.set_reconnect_interval( kReconnectIntervalMs );
    sse_.set_max_reconnect_attempts( 0 ); // unlimited
    sse_.start_async();
}

MLClientTransport::~MLClientTransport()
{
    sse_.stop(); // fast: cancels pending Get + joins
}

fastmcpp::Json MLClientTransport::request( const std::string& route, const fastmcpp::Json& payload )
{
    // Backend known dead → fail fast instead of waiting kSessionWaitTimeout for an SSE
    // session_id that won't arrive. Critical for the gateway's `tools/list` cold-start path:
    // without this, every proxied request blocks ~3 s when MI is offline, which exceeds
    // some MCP clients' `tools/list` deadline and the registry stays empty for the session.
    if ( !getBackendAlive().load() )
        throw fastmcpp::TransportError( "backend not connected" );

    std::string sid;
    {
        // Briefly wait for the SSEClient to (re)acquire a session_id. On a freshly-started
        // gateway with the backend already up, this returns immediately; on a post-`launch`
        // first call or after a backend restart, it returns as soon as the SSE on_event fires.
        std::unique_lock<std::mutex> lk( sessionMutex_ );
        sessionCv_.wait_for( lk, kSessionWaitTimeout, [this]{ return !sessionId_.empty(); } );
        sid = sessionId_;
    }
    if ( sid.empty() )
        throw fastmcpp::TransportError( "backend not connected" );

    fastmcpp::Json rpc = {
        { "jsonrpc", "2.0" },
        { "id", nextId_++ },
        { "method", route },
        { "params", payload },
    };

    auto res = httpForPost_.Post(
        ( messagesPath_ + "?session_id=" + sid ).c_str(),
        rpc.dump(),
        "application/json" );
    if ( !res )
        throw fastmcpp::TransportError(
            "POST failed: " + std::to_string( static_cast<int>( res.error() ) ) );

    if ( res->status == 404 )
    {
        // Server-side session expired. Wipe ours; SSEClient is auto-reconnecting
        // and the next endpoint event will repopulate sessionId_.
        {
            std::lock_guard<std::mutex> lk( sessionMutex_ );
            sessionId_.clear();
        }
        throw fastmcpp::TransportError( "session expired (server restarted?)" );
    }
    if ( res->status < 200 || res->status >= 300 )
        throw fastmcpp::TransportError( "HTTP " + std::to_string( res->status ) );

    fastmcpp::Json resp;
    try { resp = fastmcpp::Json::parse( res->body ); }
    catch ( const std::exception& e )
    {
        throw fastmcpp::TransportError( std::string( "POST body not JSON: " ) + e.what() );
    }

    if ( resp.contains( "error" ) )
        throw fastmcpp::Error( resp["error"].value( "message", std::string( "unknown error" ) ) );
    return resp.value( "result", fastmcpp::Json::object() );
}

} // namespace MR::McpGateway
