#pragma once

#include "MRMcp/MRFastmcpp.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>

namespace MR::McpGateway
{

/// fastmcpp transport that talks to a MeshLib-based MCP server using one
/// persistent SSE session for liveness/session-id tracking and a plain POST
/// per request, reading the JSON-RPC response from the POST body.
///
/// Sidesteps fastmcpp's per-call `SseClientTransport` pattern, whose destructor
/// blocks up to one heartbeat interval (~15 s) per call to join the listener
/// thread. Auto-reconnects on backend restart via `httplib::sse::SSEClient`.
class MLClientTransport final : public fastmcpp::client::ITransport
{
public:
    /// @param targetUrl  e.g. "http://127.0.0.1:7887". Parsed once into host+port.
    MLClientTransport( const std::string& targetUrl,
                       const std::string& ssePath,
                       const std::string& messagesPath );
    ~MLClientTransport() override;

    fastmcpp::Json request( const std::string& route, const fastmcpp::Json& payload ) override;

private:
    struct HostPort { std::string host; int port = 80; };
    static HostPort parseHostPort( const std::string& url );

    /// Delegated-to constructor: takes the already-parsed host/port so the
    /// public constructor can parse `targetUrl` exactly once.
    MLClientTransport( const HostPort& hp,
                       const std::string& ssePath,
                       const std::string& messagesPath );

private:
    // `httpForSse_` exists only as the backing `httplib::Client` that `sse_`
    // holds by reference. We never call methods on it directly after the
    // constructor configures its timeouts. We keep POSTs on a separate Client
    // so (a) read timeouts can differ — SSE wants ~5 min long-poll, POST wants
    // ~30 s — and (b) `sse_.stop()` (which internally calls `client_.stop()`)
    // doesn't yank in-flight POSTs at gateway shutdown.
    httplib::Client httpForSse_;
    httplib::sse::SSEClient sse_;
    httplib::Client httpForPost_;

    std::string messagesPath_;
    mutable std::mutex sessionMutex_;
    std::condition_variable sessionCv_;
    std::string sessionId_;
    std::atomic<int64_t> nextId_{ 1 };
};

} // namespace MR::McpGateway
