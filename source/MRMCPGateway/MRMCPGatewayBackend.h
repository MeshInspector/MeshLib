#pragma once

#include "MRMcp/MRFastmcpp.h"

#include <atomic>
#include <mutex>
#include <string>

namespace MR::McpGateway
{

struct Config;

/// Single source of truth for "is the backend currently alive?". Updated by
/// `updateBackendAliveAndNotify` (from `probeAndTrackBackend` and from the
/// persistent transport's SSE connect/error callbacks).
std::atomic<bool>& getBackendAlive();

/// Raw HTTP liveness probe. No state updates, no transition events. Used at
/// startup (before any client is connected, so transitions would be spurious).
bool probeBackendAlive( const std::string& targetUrl );

/// Updates `getBackendAlive()` and emits `notifications/tools/list_changed` on a
/// real alive↔dead transition. Suppresses the very first call (baseline-set,
/// not a transition the client cares about). Safe to call from any thread:
/// emit goes through `gatewayStdoutMutex()`.
void updateBackendAliveAndNotify( bool nowAlive );

/// Probes the backend and routes the result through `updateBackendAliveAndNotify`,
/// so a state transition synchronously emits `notifications/tools/list_changed`.
bool probeAndTrackBackend( const std::string& targetUrl );

/// Mutex held during direct gateway-side writes to stdout (notifications emitted
/// from background threads). Protects against emit-vs-emit interleaving. Does NOT
/// cover fastmcpp's own response writes — fastmcpp's stdio_server does not expose
/// its writer mutex; treat emit-vs-response interleaving as a pre-existing
/// fastmcpp limitation.
std::mutex& gatewayStdoutMutex();

/// Registers the gateway-local `launch` and `status` tools on @p proxy.
void registerLocalTools( fastmcpp::ProxyApp& proxy, const Config& cfg );

} // namespace MR::McpGateway
