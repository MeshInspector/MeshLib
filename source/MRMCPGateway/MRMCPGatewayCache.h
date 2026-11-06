#pragma once

#include <nlohmann/json.hpp>

#include <vector>

namespace MR::McpGateway
{

struct Config;

/// Tool-schema entries loaded from the on-disk cache at startup. Spliced into
/// `tools/list` responses by the gateway's handler when the backend is offline.
std::vector<nlohmann::json>& getCachedTools();

/// If the cache file is missing or its build stamp doesn't match this binary,
/// and the backend isn't already running, spawn the backend hidden so it dumps
/// its tool schemas, then embed the build stamp into the just-written file.
/// Best-effort — failures are logged to stderr but non-fatal.
void ensureFreshCache( const Config& cfg );

/// Parses `mcp_tools_cache.json` and populates `getCachedTools()`. Errors are
/// non-fatal (logged + leaves the cache empty).
void loadCachedTools( const Config& cfg );

/// Replaces the cached tools with the running backend's list and saves it if it differs,
/// so that an offline gateway lists what the backend offered last, e.g. after its update.
void updateCachedTools( const Config& cfg, const std::vector<nlohmann::json>& tools );

} // namespace MR::McpGateway
