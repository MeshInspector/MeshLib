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
    return Config::instance().getInt( cPort, 7887 );
}

void setPort( int port )
{
    Config::instance().setInt( cPort, port );
}

bool getEnableByDefault()
{
    return Config::instance().getBool( cEnableByDefault );
}

void setEnableByDefault( bool enable )
{
    Config::instance().setBool( cEnableByDefault, enable );
}

void applyToServer()
{
    Mcp::Server::Params params = Mcp::getDefaultServer().getParams();
    params.port = getPort();
    Mcp::getDefaultServer().setParams( std::move( params ) );
}

} // namespace MR::McpSettings
