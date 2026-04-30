#ifndef MESHLIB_NO_MCP

#include "MRMcp/MRMcp.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MROnInit.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRGetSystemInfoJson.h"

#include <json/json.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace MR::Mcp
{

static nlohmann::json mcpSystemLog( const nlohmann::json& args )
{
    size_t maxLines = args.value( "maxLines", uint64_t( 100 ) );
    maxLines = std::min<size_t>( maxLines, 1000 );

    const auto path = MR::Logger::instance().getLogFileName();
    if ( path.empty() )
        return nlohmann::json::object( { { "result", nlohmann::json::array() } } );

    std::ifstream in( path, std::ios::binary );
    if ( !in )
        return nlohmann::json::object( { { "result", nlohmann::json::array() } } );

    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = std::move( ss ).str();

    // Walk backwards through the file collecting the last N non-empty lines.
    std::vector<std::string> lines;
    size_t end = text.size();
    while ( end > 0 && lines.size() < maxLines )
    {
        size_t nl = text.rfind( '\n', end - 1 );
        size_t start = ( nl == std::string::npos ) ? 0 : nl + 1;
        if ( start < end )
            lines.emplace_back( text.substr( start, end - start ) );
        if ( nl == std::string::npos )
            break;
        end = nl;
    }
    std::reverse( lines.begin(), lines.end() );

    return nlohmann::json::object( { { "result", std::move( lines ) } } );
}

static nlohmann::json mcpSystemInfo( const nlohmann::json& )
{
    nlohmann::json out;
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        const Json::Value sys = MR::GetSystemInfoJson();
        Json::StreamWriterBuilder builder;
        std::unique_ptr<Json::StreamWriter> writer{ builder.newStreamWriter() };
        std::ostringstream oss;
        writer->write( sys, &oss );
        out = nlohmann::json::parse( oss.str() );
    } );
    return nlohmann::json::object( { { "result", std::move( out ) } } );
}

MR_ON_INIT{
    Server& server = getDefaultServer();

    server.addTool(
        /*id*/  "system.log",
        /*name*/"Read tail of MeshInspector log",
        /*desc*/"Return the last `maxLines` lines (default 100, capped at 1000) from MeshInspector's spdlog file. "
                "Each entry is a raw, pre-formatted line, e.g. `\"[27/04/26 16:18:53.767] [info] MCP server started "
                "on port 7887\"`. To filter by level, substring-match `[error]` / `[warn]` / `[info]` / `[debug]` / "
                "`[trace]` client-side. Returns `[]` if file logging is disabled.",
        /*input_schema*/Schema::Object{}.addMemberOpt( "maxLines", Schema::Number{} ),
        /*output_schema*/Schema::Array( Schema::String{} ),
        /*func*/mcpSystemLog
    );

    server.addTool(
        /*id*/  "system.info",
        /*name*/"Build and host environment info",
        /*desc*/"Snapshot of MeshInspector / host metadata used by the About dialog: `version`, `OS`, `CPU`, plus "
                "GPU / CUDA / framebuffer / monitor info when available. The `version` string is prefixed with "
                "`\"Debug:\"` on Debug builds. Concrete keys vary by platform/build; treat the response as a "
                "loosely-shaped JSON object.",
        /*input_schema*/Schema::Object{},
        /*output_schema*/Schema::Object{},
        /*func*/mcpSystemInfo
    );
}; // MR_ON_INIT

} // namespace MR::Mcp

#endif
