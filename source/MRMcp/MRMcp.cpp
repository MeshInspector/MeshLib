// Must not include any standard headers (fastmcpp's macro shenanigans rely on it).
#include "MRMcp/MRFastmcpp.h"

#include "MRMcp.h"

#include "MRMesh/MRBase64.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRUITestEngineControl.h"
#include "MRViewer/MRViewer.h"

#include <fstream>
#include <type_traits>


namespace MR::Mcp
{

namespace
{

// Some MCP clients serialize array/object arguments as JSON-encoded strings
// instead of native JSON values, which makes the server's argument parsing fail.
// Tracked at https://github.com/anthropics/claude-code/issues/18260 .
// For each top-level argument that the input schema types as `array` or `object`,
// reify a string-encoded value back into the proper JSON shape. Idempotent for
// well-behaved clients that already send the right type.
nlohmann::json reifyStringEncodedArgs( nlohmann::json args, const nlohmann::json& schema )
{
    if ( !args.is_object() || !schema.is_object() )
        return args;
    auto propsIt = schema.find( "properties" );
    if ( propsIt == schema.end() || !propsIt->is_object() )
        return args;
    for ( const auto& [key, propSchema] : propsIt->items() )
    {
        if ( !propSchema.is_object() )
            continue;
        auto typeIt = propSchema.find( "type" );
        if ( typeIt == propSchema.end() || !typeIt->is_string() )
            continue;
        const auto& propType = typeIt->get_ref<const std::string&>();
        if ( propType != "array" && propType != "object" )
            continue;
        auto argIt = args.find( key );
        if ( argIt == args.end() || !argIt->is_string() )
            continue;
        auto parsed = nlohmann::json::parse( argIt->get_ref<const std::string&>(), nullptr, /* allow_exceptions */ false );
        if ( !parsed.is_discarded() )
            *argIt = std::move( parsed );
    }
    return args;
}

} // namespace

struct Server::State
{
    fastmcpp::tools::ToolManager toolManager; // This has to be persistent, or `fastmcpp::mcp::make_mcp_handler()` dangles it.
    std::unordered_map<std::string, std::string> toolDescs; // No idea why this is not a part of `toolManager`.
    std::optional<fastmcpp::server::SseServerWrapper> server;
    Server::ToolValidator toolValidator; // Empty by default; consulted per tool call when set.

    void createServer( const Params& params );

    // Build the catalog entry JSON for a single registered tool. Shared by
    // `dumpToolsAsJson()` and the `/api/tool/<id>` HTTP route.
    nlohmann::json buildToolEntryJson( const std::string& name ) const;

    // HTTP handlers for the `/api/*` REST routes registered in createServer().
    void handleToolsList( const httplib::Request& req, httplib::Response& res );
    void handleToolDescribe( const httplib::Request& req, httplib::Response& res );
    void handleToolsCall( const httplib::Request& req, httplib::Response& res, bool isGet );
};

void Server::State::createServer( const Params& params )
{
    assert( !server );
    server.emplace( fastmcpp::mcp::make_mcp_handler( params.name, params.version, toolManager, toolDescs ), params.address, params.port );

    auto onList = [this]( const httplib::Request& req, httplib::Response& res ) {
        handleToolsList( req, res );
    };
    server->add_route( "GET",  "/api/tools/list", onList );
    server->add_route( "POST", "/api/tools/list", onList );

    auto onDescribe = [this]( const httplib::Request& req, httplib::Response& res ) {
        handleToolDescribe( req, res );
    };
    server->add_route( "GET",  R"(/api/tool/(.+))", onDescribe );
    server->add_route( "POST", R"(/api/tool/(.+))", onDescribe );

    auto onCall = [this]( const httplib::Request& req, httplib::Response& res ) {
        handleToolsCall( req, res, /* isGet = */ req.method == "GET" );
    };
    server->add_route( "GET",  R"(/api/tools/call/(.+))", onCall );
    server->add_route( "POST", R"(/api/tools/call/(.+))", onCall );
}

nlohmann::json Server::State::buildToolEntryJson( const std::string& name ) const
{
    const auto& tool = toolManager.get( name );
    nlohmann::json entry = {
        { "name", tool.name() },
        { "inputSchema", tool.input_schema() },
    };
    if ( tool.title().has_value() )
        entry["title"] = *tool.title();
    if ( tool.description().has_value() )
        entry["description"] = *tool.description();
    else if ( auto it = toolDescs.find( name ); it != toolDescs.end() )
        entry["description"] = it->second;
    const auto& outSchema = tool.output_schema();
    if ( !outSchema.is_null() && !( outSchema.is_object() && outSchema.empty() ) )
    {
        // MCP requires `outputSchema.type == "object"`. fastmcpp's mcp/handler.cpp
        // wraps non-object schemas at runtime; mirror the same shape here so
        // cached entries match what the live server emits in `tools/list`.
        const bool alreadyObject = outSchema.is_object()
            && outSchema.contains( "type" ) && outSchema.at( "type" ) == "object";
        if ( alreadyObject )
            entry["outputSchema"] = outSchema;
        else
            entry["outputSchema"] = {
                { "type", "object" },
                { "properties", { { "result", outSchema } } },
                { "required", nlohmann::json::array( { "result" } ) },
                { "x-fastmcp-wrap-result", true },
            };
    }
    return entry;
}

namespace
{

// JSON-RPC 2.0 standard error codes (https://www.jsonrpc.org/specification#error_object).
enum class RpcErrorCode : int
{
    ParseError     = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams  = -32602,
    InternalError  = -32603,
};

void writeJsonError( httplib::Response& res, int httpStatus, std::string message, RpcErrorCode rpcCode )
{
    res.status = httpStatus;
    res.set_content( nlohmann::json{
        { "error", std::move( message ) },
        { "code",  static_cast<int>( rpcCode ) },
    }.dump(), "application/json" );
}

// Inspect a tool's raw output for the file-output convention `{bytes, contentType}`,
// transparently unwrapping one level of `{"result": {...}}` (the existing wrap used
// by tools in MRViewerMcp / MRUiMcp / MRSystemMcp / MRSceneMcp).
//
// Returns a pointer to the inner object that has both `bytes` and `contentType`,
// or `nullptr` if the convention isn't matched.
const nlohmann::json* matchFileOutput( const nlohmann::json& result )
{
    auto isFileShape = []( const nlohmann::json& j ) {
        return j.is_object()
            && j.contains( "bytes" )       && j["bytes"].is_string()
            && j.contains( "contentType" ) && j["contentType"].is_string();
    };
    if ( isFileShape( result ) )
        return &result;
    if ( result.is_object() && result.contains( "result" ) && isFileShape( result["result"] ) )
        return &result["result"];
    return nullptr;
}

} // namespace

void Server::State::handleToolsList( const httplib::Request&, httplib::Response& res )
{
    auto tools = nlohmann::json::array();
    for ( const auto& name : toolManager.list_names() )
        tools.push_back( buildToolEntryJson( name ) );
    res.status = 200;
    res.set_content( nlohmann::json{ { "tools", std::move( tools ) } }.dump(), "application/json" );
}

void Server::State::handleToolDescribe( const httplib::Request& req, httplib::Response& res )
{
    const std::string toolId = req.matches.size() > 1 ? req.matches[1].str() : std::string{};
    if ( !toolManager.has( toolId ) )
    {
        writeJsonError( res, 404, "tool not found: " + toolId, RpcErrorCode::MethodNotFound );
        return;
    }
    res.status = 200;
    res.set_content( buildToolEntryJson( toolId ).dump(), "application/json" );
}

void Server::State::handleToolsCall( const httplib::Request& req, httplib::Response& res, bool isGet )
{
    const std::string toolId = req.matches.size() > 1 ? req.matches[1].str() : std::string{};
    if ( !toolManager.has( toolId ) )
    {
        writeJsonError( res, 404, "tool not found: " + toolId, RpcErrorCode::MethodNotFound );
        return;
    }

    nlohmann::json args = nlohmann::json::object();
    if ( isGet )
    {
        // URL query params become tool arguments. Each value is JSON-parsed when
        // possible (so `?width=1920` -> 1920, `?path=[]` -> [], `?on=true` -> true)
        // and falls back to the raw string when not (`?label=foo` -> "foo"). Tool
        // schema validation still runs in toolManager.invoke and rejects type
        // mismatches with RpcErrorCode::InvalidParams.
        for ( const auto& [k, v] : req.params )
        {
            auto parsed = nlohmann::json::parse( v, nullptr, /* allow_exceptions */ false );
            args[k] = parsed.is_discarded() ? nlohmann::json( v ) : std::move( parsed );
        }
    }
    else if ( !req.body.empty() )
    {
        auto parsed = nlohmann::json::parse( req.body, nullptr, /* allow_exceptions */ false );
        if ( parsed.is_discarded() )
        {
            writeJsonError( res, 400, "invalid JSON request body", RpcErrorCode::ParseError );
            return;
        }
        args = std::move( parsed );
    }

    nlohmann::json result;
    try
    {
        result = toolManager.invoke( toolId, args );
    }
    catch ( const fastmcpp::NotFoundError& e )
    {
        writeJsonError( res, 404, e.what(), RpcErrorCode::MethodNotFound );
        return;
    }
    catch ( const fastmcpp::ValidationError& e )
    {
        writeJsonError( res, 400, e.what(), RpcErrorCode::InvalidParams );
        return;
    }
    catch ( const std::exception& e )
    {
        writeJsonError( res, 500, e.what(), RpcErrorCode::InternalError );
        return;
    }

    if ( const auto* file = matchFileOutput( result ); file )
    {
        auto bytes = decode64( ( *file )["bytes"].get<std::string>() );
        const auto& contentType = ( *file )["contentType"].get_ref<const std::string&>();
        res.status = 200;
        res.set_content(
            std::string( reinterpret_cast<const char*>( bytes.data() ), bytes.size() ),
            contentType.c_str() );
        return;
    }

    res.status = 200;
    res.set_content( result.dump(), "application/json" );
}

Server::Params::Params()
    : name( getProductName() ),
    version( GetMRVersionString() )
{}

Server::Server() = default;
Server::Server( Server&& ) = default;
Server& Server::operator=( Server&& ) = default;
Server::~Server() = default;

bool Server::addTool( std::string id, std::string name, std::string desc, Schema::Base inputSchema, Schema::Base outputSchema, ToolFunc func )
{
    if ( !state_ )
    {
        state_ = std::make_unique<State>();
    }
    else if ( state_->server )
    {
        assert( false && "`MR::Mcp::Server::addTool()`: Called too late, the server is already initialized." );
        return false;
    }

    // Why is managing the descriptions not automated by the library?
    if ( !state_->toolDescs.try_emplace( id, desc ).second )
    {
        assert( false && "`MR::Mcp::Server::addTool()`: Duplicate tool id." );
        return false;
    }

    auto schemaJson = std::move( inputSchema ).asJson();
    auto wrapped = [statePtr = state_.get(), id, schema = schemaJson, inner = std::move( func )]( const nlohmann::json& args ) -> nlohmann::json
    {
        if ( statePtr->toolValidator )
        {
            if ( auto res = statePtr->toolValidator( id ); !res )
                throw std::runtime_error( std::move( res.error() ) );
        }
        return inner( reifyStringEncodedArgs( args, schema ) );
    };
    state_->toolManager.register_tool( fastmcpp::tools::Tool( id, std::move( schemaJson ), std::move( outputSchema ).asJson(), std::move( wrapped ), name, desc, {} ) );
    return true;
}

const Server::Params& Server::getParams() const
{
    return params_;
}

void Server::setParams( Server::Params params )
{
    if ( params_ == params )
        return;

    const bool serverExisted = state_ && bool( state_->server );
    const bool serverWasRunning = serverExisted && isRunning();

    if ( serverWasRunning )
        setRunning( false );
    if ( serverExisted )
        state_->server.reset();

    params_ = std::move( params );

    if ( serverExisted )
        state_->createServer( params );
    if ( serverWasRunning )
        setRunning( true );
}

bool Server::isRunning() const
{
    return state_ && state_->server && state_->server->running();
}

bool Server::setRunning( bool enable )
{
    if ( enable )
    {
        if ( !state_ )
            state_ = std::make_unique<State>();
        if ( !state_->server )
            state_->createServer( params_ );

        bool ok = state_->server->start();
        if ( ok )
            spdlog::info( "MCP server started on port {}", getParams().port );
        else
            spdlog::error( "MCP server failed to start on port {}", getParams().port );
        return ok;
    }
    else
    {
        if ( state_ && state_->server )
        {
            state_->server->stop();
            spdlog::info( "MCP server stopped" );
        }
        return true;
    }
}

void Server::shutdown()
{
    state_.reset();
}

nlohmann::json Server::dumpToolsAsJson() const
{
    auto out = nlohmann::json::array();
    if ( !state_ )
        return out;
    for ( const auto& name : state_->toolManager.list_names() )
        out.push_back( state_->buildToolEntryJson( name ) );
    return out;
}

Expected<void> Server::saveToolsCache( const std::filesystem::path& path ) const
{
    nlohmann::json envelope = { { "tools", dumpToolsAsJson() } };

    std::error_code ec;
    if ( !path.parent_path().empty() )
    {
        std::filesystem::create_directories( path.parent_path(), ec );
        if ( ec )
            return unexpected( fmt::format( "cannot create directory {}: {}",
                utf8string( path.parent_path() ), ec.message() ) );
    }

    // Write to a sibling .tmp first then rename: this is atomic on the filesystem,
    // so a concurrent reader (e.g. the gateway polling for the cache) never sees a
    // partial write or an empty file mid-flush.
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream f( tmp );
        if ( !f )
            return unexpected( fmt::format( "cannot open {} for writing", utf8string( tmp ) ) );
        f << envelope.dump( 2 );
    }
    std::filesystem::rename( tmp, path, ec );
    if ( ec )
    {
        std::error_code removeTempEc;
        std::filesystem::remove( tmp, removeTempEc );
        return unexpected( fmt::format( "cannot rename {} -> {}: {}",
            utf8string( tmp ), utf8string( path ), ec.message() ) );
    }
    spdlog::info( "MRMcp: dumped {} tools to {}", state_ ? state_->toolManager.list_names().size() : 0u, utf8string( path ) );
    return {};
}

void Server::setToolValidator( ToolValidator validator )
{
    if ( !state_ )
        state_ = std::make_unique<State>();
    state_->toolValidator = std::move( validator );
}

Server& getDefaultServer()
{
    static Server ret;
    return ret;
}

} // namespace MR
