// Must not include any standard headers (fastmcpp's macro shenanigans rely on it).
#include "MRMcp/MRFastmcpp.h"

#include "MRMcp.h"

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

    void createServer( const Params& params )
    {
        assert( !server );
        server.emplace( fastmcpp::mcp::make_mcp_handler( params.name, params.version, toolManager, toolDescs ), params.address, params.port );
    }
};

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
    {
        const auto& tool = state_->toolManager.get( name );
        nlohmann::json entry = {
            { "name", tool.name() },
            { "inputSchema", tool.input_schema() },
        };
        if ( tool.title().has_value() )
            entry["title"] = *tool.title();
        if ( tool.description().has_value() )
            entry["description"] = *tool.description();
        else if ( auto it = state_->toolDescs.find( name ); it != state_->toolDescs.end() )
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
        out.push_back( std::move( entry ) );
    }
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
