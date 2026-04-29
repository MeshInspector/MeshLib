#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace MR::McpGateway
{

/// Aggregates all values parsed from the gateway's command-line. Passed by const-ref
/// to the cache, backend-probe, and local-tool-registration helpers — letting them
/// read what they need without each replicating its own subset of CLI parsing.
struct Config
{
    int mcpPort              = 7887;                   ///< -mcpPort N forwarded to spawned MI; targetUrl is derived from this if --target-url is omitted.
    std::string targetUrl    = "http://127.0.0.1:7887";
    std::string ssePath      = "/sse";
    std::string messagesPath = "/messages";
    std::filesystem::path launchCommand;
    std::vector<std::string> launchArgs;
    std::chrono::seconds launchTimeout{ 30 };
    std::string toolsCacheNamespace; ///< --tools-cache-namespace <name> (optional sub-folder)
};

} // namespace MR::McpGateway
