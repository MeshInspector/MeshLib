#pragma once

#include "exports.h"

// Those functions act on the default MCP settings in the config file.

namespace MR::McpSettings
{

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
