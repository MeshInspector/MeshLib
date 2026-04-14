#pragma once

#include "exports.h"

#include <memory>

namespace MR
{

class McpServer
{
    struct State;

    int port_ = -1; // See the constructor.
    std::unique_ptr<State> state_;

public:
    MRVIEWER_API McpServer();
    MRVIEWER_API McpServer( McpServer&& );
    MRVIEWER_API McpServer& operator=( McpServer&& );
    MRVIEWER_API ~McpServer();

    [[nodiscard]] int getPort() const { return port_; }
    void setPort( int port ) { port_ = port; }

    // This stops the server and applies the new settings, such as the port.
    // Then call `setRunning( true )` to start it again.
    MRVIEWER_API void recreateServer();

    [[nodiscard]] MRVIEWER_API bool isRunning() const;
    // Returns true on success, including if the server is already running and you're trying to start it again.
    // Stopping always returns true.
    MRVIEWER_API bool setRunning( bool enable );
};

[[nodiscard]] MRVIEWER_API McpServer& getDefaultMcpServer();

} // namespace MR
