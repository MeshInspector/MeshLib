#pragma once

#include "exports.h"

#include <filesystem>
#include <string>
#include <vector>

// Those functions act on the default MCP settings in the config file.

namespace MR::McpSettings
{

// Overrides parsed from the application's command-line arguments. Sentinel values
// (`port <= 0`, empty `dumpFilePath`) mean "no override".
struct CmdLineOverrides
{
    int port = 0;                        ///< `-mcpPort N`. <= 0 means no override.
    std::filesystem::path dumpFilePath;  ///< `-mcpDumpFile <path>`. Empty means no dump.
};

// Pure parse: scans @p commandArgs for MCP-related flags and returns the resolved
// overrides. Last occurrence of each flag wins (matches shell convention).
// `-mcpPort N` forces the server port to N (overriding the config).
// `-mcpDumpFile <path>` requests writing the tool cache to that path; the caller
// (typically `ViewerSetup::setupMcp`) is expected to skip starting the live server
// in that case so a prime spawn does not collide with a real backend on the port.
[[nodiscard]] MRVIEWER_API CmdLineOverrides parseCmdLineOverrides( const std::vector<std::string>& commandArgs );

// Returns the MCP port from the config file, or the default value.
// Note that this acts on the config file and not on the actual MCP server that might be running.
[[nodiscard]] MRVIEWER_API int getPort();

// Sets the MCP port in the config.
// Note that this acts on the config file and not on the actual MCP server that might be running.
MRVIEWER_API void setPort( int port );

// Returns whether the MCP server should start on program startup, according to the config file.
[[nodiscard]] MRVIEWER_API bool getEnableByDefault();

// Sets whether the MCP server should start on program startup, in the config file.
MRVIEWER_API void setEnableByDefault( bool enable );

// Sends those settings to the MCP server. This restarts it if necessary, e.g. to update the port.
// This ignores `getEnableByDefault()`.
MRVIEWER_API void applyToServer();

// True iff `-mcpPort N` was passed on the command line. The config-backed port is then
// ignored for this session, and the GUI shows the port read-only.
[[nodiscard]] MRVIEWER_API bool isPortLockedFromCmdLine();

} // namespace MR::McpSettings
