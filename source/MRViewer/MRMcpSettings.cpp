#include "MRMcpSettings.h"

#include "MRMcp/MRMcp.h"
#include "MRMesh/MRConfig.h"

#include <utility>

namespace MR::McpSettings
{

static const std::string cPort = "mcp.port";
static const std::string cEnableByDefault = "mcp.enableByDefault";

int getPort()
{
    #ifndef MESHLIB_NO_MCP
    return Config::instance().getInt( cPort, 7887 );
    #else
    return -1;
    #endif
}

void setPort( int port )
{
    #ifndef MESHLIB_NO_MCP
    Config::instance().setInt( cPort, port );
    #endif
}

bool getEnableByDefault()
{
    #ifndef MESHLIB_NO_MCP
    return Config::instance().getBool( cEnableByDefault );
    #else
    return false;
    #endif
}

void setEnableByDefault( bool enable )
{
    #ifndef MESHLIB_NO_MCP
    Config::instance().setBool( cEnableByDefault, enable );
    #endif
}

void applyToServer()
{
    #ifndef MESHLIB_NO_MCP
    Mcp::Server::Params params = Mcp::getDefaultServer().getParams();
    params.port = getPort();
    Mcp::getDefaultServer().setParams( std::move( params ) );
    #endif
}

} // namespace MR::McpSettings
