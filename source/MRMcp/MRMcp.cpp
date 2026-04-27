// Must not include any standard headers

#undef _t // Our translation macro interefers with Fastmcpp.

#if defined( __GNUC__ )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined( _MSC_VER )
#pragma warning( push )
#pragma warning( disable: 4100 ) // unreferenced formal parameter
#pragma warning( disable: 4355 ) // 'this': used in base member initializer list
#endif

// This must be included before any standard library headers, because of the macro shenanigans we added to that header.
// Those are duplicated into our PCH, so that shouldn't interfere.
#include <fastmcpp.hpp>
#include <fastmcpp/server/sse_server.hpp>

#if defined( __GNUC__ )
#pragma GCC diagnostic pop
#elif defined( _MSC_VER )
#pragma warning( pop )
#endif

#include "MRMcp.h"

#include "MRMesh/MRSystem.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRUITestEngineControl.h"
#include "MRViewer/MRViewer.h"

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
