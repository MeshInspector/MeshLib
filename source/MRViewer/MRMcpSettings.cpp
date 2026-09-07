#include "MRMcpSettings.h"

#include "MRMesh/MRConfig.h"
#include "MRMesh/MRStringConvert.h"

#ifndef MESHLIB_NO_MCP
#include "MRMcp/MRMcp.h"
#include "MRViewer/MRViewer.h"
#endif

#include <utility>

namespace MR::McpSettings
{

#ifndef MESHLIB_NO_MCP
static const std::string cPort = "mcp.port";
static const std::string cEnableByDefault = "mcp.enableByDefault";
#endif

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
    #else
    (void)port;
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
    #else
    (void)enable;
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

bool isPortLockedFromCmdLine()
{
    #ifndef MESHLIB_NO_MCP
    static const bool locked = []
    {
        return parseCmdLineOverrides( getViewerInstance().commandArgs ).port > 0;
    }();
    return locked;
    #else
    return false;
    #endif
}

CmdLineOverrides parseCmdLineOverrides( const std::vector<std::string>& commandArgs )
{
    CmdLineOverrides out;
    for ( size_t i = 0; i + 1 < commandArgs.size(); ++i )
    {
        if ( commandArgs[i] == "-mcpPort" )
            out.port = std::atoi( commandArgs[i + 1].c_str() );
        else if ( commandArgs[i] == "-mcpDumpFile" )
            out.dumpFilePath = pathFromUtf8( commandArgs[i + 1] );
    }
    return out;
}

} // namespace MR::McpSettings
