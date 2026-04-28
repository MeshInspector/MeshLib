#pragma once

#include "exports.h"

// Those functions act on the default MCP settings in the config file.

namespace MR::McpSettings
{

[[nodiscard]] MRVIEWER_API int getPort();
MRVIEWER_API void setPort( int port );

[[nodiscard]] MRVIEWER_API bool getEnableByDefault();
MRVIEWER_API void setEnableByDefault( bool enable );

// Sends those settings to the MCP server. This restarts it if necessary, e.g. to update the port.
// This ignores `getEnableByDefault()`.
MRVIEWER_API void applyToServer();

} // namespace MR::McpSettings
