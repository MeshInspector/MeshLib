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

struct Server::State
{
    Params params;
    fastmcpp::tools::ToolManager toolManager; // This has to be persistent, or `fastmcpp::mcp::make_mcp_handler()` dangles it.
    std::unordered_map<std::string, std::string> toolDescs; // No idea why this is not a part of `toolManager`.
    std::optional<fastmcpp::server::SseServerWrapper> server;

    void createServer()
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

    state_->toolManager.register_tool( fastmcpp::tools::Tool( id, std::move( inputSchema ).asJson(), std::move( outputSchema ).asJson(), func, name, desc, {} ) );
    return true;
}

Server::Params Server::getParams() const
{
    if ( state_ )
        return state_->params;
    else
        return {};
}

void Server::setParams( Server::Params params )
{
    const bool serverExisted = state_ && bool( state_->server );
    const bool serverWasRunning = serverExisted && isRunning();
    if ( serverWasRunning )
        setRunning( false );

    state_->server.reset();
    state_->params = std::move( params );

    if ( serverExisted )
        state_->createServer();
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
            state_->createServer();

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

Server& getDefaultServer()
{
    static Server ret;
    return ret;
}

} // namespace MR
