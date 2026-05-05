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

} // namespace MR::McpGateway
