#pragma once

#include "MRMcp/MRFastmcpp.h"

#include <atomic>
#include <string>

namespace MR::McpGateway
{

struct Config;

/// Single source of truth for "is the backend currently alive?". Updated by every
/// `probeAndTrackBackend` call.
extern std::atomic<bool> g_backendAlive;

/// Raw HTTP liveness probe. No state updates, no transition events. Used at
/// startup (before any client is connected, so transitions would be spurious).
bool probeBackendAlive( const std::string& targetUrl );

/// Probes the backend, updates `g_backendAlive`, and on a state transition
/// synchronously emits `notifications/tools/list_changed` so the connected MCP
/// client surfaces / drops the proxied tool set without polling.
bool probeAndTrackBackend( const std::string& targetUrl );

/// Registers the gateway-local `launch` and `status` tools on @p proxy.
void registerLocalTools( fastmcpp::ProxyApp& proxy, const Config& cfg );

} // namespace MR::McpGateway
