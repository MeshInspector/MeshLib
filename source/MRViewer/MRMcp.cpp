#include "MRMcp.h"

#if MR_ENABLE_MCP_SERVER
#include "MRMesh/MRSystem.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRUITestEngineControl.h"
#include "MRViewer/MRViewer.h"

#undef _t // Our translation macro interefers with Fastmcpp.

#if defined( __GNUC__ )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined( _MSC_VER )
#pragma warning( push )
#pragma warning( disable: 4100 ) // unreferenced formal parameter
#pragma warning( disable: 4355 ) // 'this': used in base member initializer list
#endif

#include <fastmcpp.hpp>
#include <fastmcpp/server/sse_server.hpp>

#if defined( __GNUC__ )
#pragma GCC diagnostic pop
#elif defined( _MSC_VER )
#pragma warning( pop )
#endif

#include <type_traits>
#endif


/* HOW TO TEST MCP:

Use "MCP Inspector". Run it with `npx @modelcontextprotocol/inspector`, where `npx` is installed as a part of Node.JS.
Set:
    Transport Type = SSE
    URL = http://localhost:8080/sse
    Connection Type = Via Proxy  (Doesn't work for me without proxy now when we're using the Fastmcpp library, but did work with another library; not sure why.)

Press `Connect`.
Press `List Tools` (if grayed out, do `Clear` first).
Click on your tool.
On the right panel, set parameters.
    For some parameter types, it helps to press `Switch to JSON` on the right, then type them as JSON.
Press `Run Tool`.
    Note, you might need to press this twice. In some cases, the first press passes stale/empty parameters.
Then check for validation errors, below this button.

If your output doesn't match the schema you specified, paste both the output and the schema (using the `Copy` button in the top-right corner of the code blocks; that copies JSON properly, unlike Ctrl+C in this case)
  into a schema validator, e.g. https://www.jsonschemavalidator.net/

*/


namespace MR
{

struct McpServer::State
{
    #if MR_ENABLE_MCP_SERVER
    fastmcpp::tools::ToolManager tool_manager; // This has to be persistent, or `fastmcpp::mcp::make_mcp_handler()` dangles it.
    fastmcpp::server::SseServerWrapper server;


    State( int port, std::string name, std::string version, fastmcpp::tools::ToolManager new_tool_manager, const std::unordered_map<std::string, std::string>& tool_descs )
        : tool_manager( std::move( new_tool_manager ) ),
        server( fastmcpp::mcp::make_mcp_handler( name, version, tool_manager, tool_descs ), "127.0.0.1", port )
    {}
    #endif
};

McpServer::McpServer()
    : port_( 7887 ) // An arbirary value.
{
    recreateServer();
}

McpServer::McpServer( McpServer&& ) = default;
McpServer& McpServer::operator=( McpServer&& ) = default;
McpServer::~McpServer() = default;

void McpServer::recreateServer()
{
    #if MR_ENABLE_MCP_SERVER

    fastmcpp::tools::ToolManager tool_manager;
    std::unordered_map<std::string, std::string> tool_descs;

    auto addTool = [&]( std::string id, std::string name, std::string desc, fastmcpp::Json input_schema, fastmcpp::Json output_schema, fastmcpp::tools::Tool::Fn func )
    {
        tool_descs.try_emplace( id, desc ); // Why is this not automated by the library?
        tool_manager.register_tool( fastmcpp::tools::Tool( id, input_schema, output_schema, func, name, desc, {} ) );
    };

    static const auto skipFramesAfterInput = []
    {
        for ( int i = 0; i < MR::getViewerInstance().forceRedrawMinimumIncrementAfterEvents; ++i )
            MR::CommandLoop::runCommandFromGUIThread( [] {} ); // Wait a few frames.
    };

    addTool(
        /*id*/"ui.listEntries",
        /*name*/"List UI entries",
        /*desc*/"Returns the list of UI elements at the given path. The elements form a tree. Pass an empty array to get the top-level elements. Each element is described by a string. The path parameter describes the path from the root node to a specific element. Only elements of type `group` can have sub-elements.",
        /*input_schema*/fastmcpp::Json::object( {
            { "type", "object" },
            { "properties", fastmcpp::Json::object( {
                { "path", fastmcpp::Json::object( {
                    { "type", "array" },
                    { "items", fastmcpp::Json::object( {
                        { "type", "number" },
                    } ) },
                } ) }
            } ) },
            { "required", fastmcpp::Json::array( {
                "path",
            } ) },
        } ),
        /*output_schema*/fastmcpp::Json::object( {
            { "type", "array" },
            { "items", fastmcpp::Json::object( {
                { "type", "object" },
                { "properties", fastmcpp::Json::object( {
                    { "name", fastmcpp::Json::object( {
                        { "type", "string" },
                    } ) },
                    { "type", fastmcpp::Json::object( {
                        { "type", "string" },
                    } ) },
                } ) },
                { "required", fastmcpp::Json::array( {
                    "name",
                    "type",
                } ) },
            } ) },
        } ),
        /*func*/[]( const fastmcpp::Json& params ) -> fastmcpp::Json
        {
            if ( !params.contains( "path" ) )
                throw std::runtime_error( "The path parameter is missing." );

            std::vector<UI::TestEngine::Control::TypedEntry> list;
            MR::CommandLoop::runCommandFromGUIThread( [&]
            {
                auto ex = UI::TestEngine::Control::listEntries( params["path"].get<std::vector<std::string>>() );
                if ( !ex )
                    throw std::runtime_error( ex.error() );
                list = std::move( *ex );
            } );

            fastmcpp::Json ret = fastmcpp::Json::array();
            for ( const auto& elem : list )
            {
                std::string typeStr;
                switch ( elem.type )
                {
                    case UI::TestEngine::Control::EntryType::button:      typeStr = "button"; break;
                    case UI::TestEngine::Control::EntryType::group:       typeStr = "group"; break;
                    case UI::TestEngine::Control::EntryType::valueInt:    typeStr = "int"; break;
                    case UI::TestEngine::Control::EntryType::valueUint:   typeStr = "uint"; break;
                    case UI::TestEngine::Control::EntryType::valueReal:   typeStr = "float"; break; // Hopefully "float" is more clear to LLMs than "real". The actual underlying type is `double`.
                    case UI::TestEngine::Control::EntryType::valueString: typeStr = "string"; break;
                }

                assert( !typeStr.empty() );
                if ( typeStr.empty() )
                    typeStr = "invalid";

                ret.push_back( fastmcpp::Json::object( {
                    { "name", elem.name },
                    { "type", std::move( typeStr ) },
                } ) );
            }

            return fastmcpp::Json::object( { { "result", ret } } );
        }
    );

    addTool(
        /*id*/"ui.pressButton",
        /*name*/"Press button",
        /*desc*/"Presses the button at the given path.",
        /*input_schema*/fastmcpp::Json::object( {
            { "type", "object" },
            { "properties", fastmcpp::Json::object( {
                { "path", fastmcpp::Json::object( {
                    { "type", "array" },
                    { "items", fastmcpp::Json::object( {
                        { "type", "number" },
                    } ) },
                } ) },
            } ) },
            { "required", fastmcpp::Json::array( {
                "path",
            } ) },
        } ),
        /*output_schema*/fastmcpp::Json::object(),
        /*func*/[]( const fastmcpp::Json& params ) -> fastmcpp::Json
        {
            if ( !params.contains( "path" ) )
                throw std::runtime_error( "The path parameter is missing." );

            MR::CommandLoop::runCommandFromGUIThread( [&]
            {
                auto ex = UI::TestEngine::Control::pressButton( params["path"].get<std::vector<std::string>>() );
                if ( !ex )
                    throw std::runtime_error( ex.error() );
            } );
            skipFramesAfterInput();

            return fastmcpp::Json::object();
        }
    );

    auto handleValueType = [&]<typename T>( const std::string& typeName )
    {
        addTool(
            /*id*/"ui.readValue" + typeName,
            /*name*/"Read " + typeName + " value",
            /*desc*/"Reads the value at the given path, of type `" + typeName + "`." +
            (
                std::is_same_v<T, std::string>
                ?
                    " If the result contains an array called `allowedValues`, then when assigning a new value using `ui.writeValue" + typeName + "`, it must match one of the strings listed in `allowedValues`."
                :
                    " When assigning a new value using `ui.writeValue" + typeName + "`, it must be between `min` and `max` inclusive."
            ),
            /*input_schema*/fastmcpp::Json::object( {
                { "type", "object" },
                { "properties", fastmcpp::Json::object( {
                    { "path", fastmcpp::Json::object( {
                        { "type", "array" },
                        { "items", fastmcpp::Json::object( {
                            { "type", "number" },
                        } ) },
                    } ) },
                } ) },
                { "required", fastmcpp::Json::array( {
                    "path",
                } ) },
            } ),
            /*output_schema*/(
                std::is_same_v<T, std::string>
                ?
                    fastmcpp::Json::object( {
                        { "type", "object" },
                        { "properties", fastmcpp::Json::object( {
                            { "value", fastmcpp::Json::object( {
                                { "type", "string" },
                            } ) },
                            { "allowedValues", fastmcpp::Json::object( {
                                { "type", "array" },
                                { "items", fastmcpp::Json::object( {
                                    { "type", "string" },
                                } ) },
                            } ) },
                        } ) },
                        { "required", fastmcpp::Json::array( {
                            "value",
                        } ) },
                    } )
                :
                    fastmcpp::Json::object( {
                        { "type", "object" },
                        { "properties", fastmcpp::Json::object( {
                            { "value", fastmcpp::Json::object( {
                                { "type", "number" },
                            } ) },
                            { "min", fastmcpp::Json::object( {
                                { "type", "number" },
                            } ) },
                            { "max", fastmcpp::Json::object( {
                                { "type", "number" },
                            } ) },
                        } ) },
                        { "required", fastmcpp::Json::array( {
                            "value",
                            "min",
                            "max",
                        } ) },
                    } )
            ),
            /*func*/[]( const fastmcpp::Json& params ) -> fastmcpp::Json
            {
                if ( !params.contains( "path" ) )
                    throw std::runtime_error( "The path parameter is missing." );

                UI::TestEngine::Control::Value<T> value;
                MR::CommandLoop::runCommandFromGUIThread( [&]
                {
                    auto ex = UI::TestEngine::Control::readValue<T>( params["path"].get<std::vector<std::string>>() );
                    if ( !ex )
                        throw std::runtime_error( ex.error() );
                    value = std::move( *ex );
                } );

                fastmcpp::Json ret = fastmcpp::Json::object();
                ret["value"] = value.value;
                if constexpr ( std::is_same_v<T, std::string> )
                {
                    if ( value.allowedValues )
                        ret["allowedValues"] = *value.allowedValues;
                }
                else
                {
                    ret["min"] = value.min;
                    ret["max"] = value.max;
                }

                return ret;
            }
        );

        addTool(
            /*id*/"ui.writeValue" + typeName,
            /*name*/"Write " + typeName + " value",
            /*desc*/"Writes the value at the given path, of type `" + typeName + "`. You can call `ui.readValue" + typeName + "` before this to know what values are allowed.",
            /*input_schema*/fastmcpp::Json::object( {
                { "type", "object" },
                { "properties", fastmcpp::Json::object( {
                    { "path", fastmcpp::Json::object( {
                        { "type", "array" },
                        { "items", fastmcpp::Json::object( {
                            { "type", "number" },
                        } ) },
                    } ) },
                    { "value", fastmcpp::Json::object( {
                        { "type", "number" },
                    } ) },
                } ) },
                { "required", fastmcpp::Json::array( {
                    "path",
                    "value",
                } ) },
            } ),
            /*output_schema*/fastmcpp::Json::object(),
            /*func*/[]( const fastmcpp::Json& params ) -> fastmcpp::Json
            {
                if ( !params.contains( "path" ) )
                    throw std::runtime_error( "The path parameter is missing." );

                if ( !params.contains( "path" ) )
                    throw std::runtime_error( "The path parameter is missing." );
                if ( !params.contains( "value" ) )
                    throw std::runtime_error( "The value parameter is missing." );

                MR::CommandLoop::runCommandFromGUIThread( [&]
                {
                    auto ex = UI::TestEngine::Control::writeValue<T>( params["path"].get<std::vector<std::string>>(), T( params["value"] ) );
                    if ( !ex )
                        throw std::runtime_error( ex.error() );
                } );
                skipFramesAfterInput();

                return fastmcpp::Json::object();
            }
        );
    };

    handleValueType.operator()<std::int64_t>( "Int" );
    handleValueType.operator()<std::uint64_t>( "Uint" );
    handleValueType.operator()<double>( "Real" );
    handleValueType.operator()<std::string>( "String" );

    state_ = std::make_unique<State>( port_, "MeshInspector", GetMRVersionString(), std::move( tool_manager ), tool_descs );

    #endif
}

bool McpServer::isRunning() const
{
    #if MR_ENABLE_MCP_SERVER
    return state_->server.running();
    #else
    return false;
    #endif
}

bool McpServer::setRunning( bool enable )
{
    #if MR_ENABLE_MCP_SERVER
    if ( enable )
    {
        bool ok = state_->server.start();
        if ( ok )
            spdlog::info( "MCP server started on port {}", getPort() );
        else
            spdlog::error( "MCP server failed to start on port {}", getPort() );
        return ok;
    }
    else
    {
        state_->server.stop();
        spdlog::info( "MCP server stopped" );
        return true;
    }
    #else
    (void)enable;
    return false;
    #endif
}

McpServer& getDefaultMcpServer()
{
    static McpServer ret = []
    {
        McpServer server;
        server.setRunning( true );
        return server;
    }();
    return ret;
}

#if MR_ENABLE_MCP_SERVER
static const std::nullptr_t init_mcp = []{
    // Poke the default MCP server to start it.
    // Use `CommandLoop` to delay initialization until the viewer finishes initializing.
    // Otherwise this gets called too early, before even the logger is configured.
    CommandLoop::appendCommand( []{ (void)getDefaultMcpServer(); } );

    return nullptr;
}();
#endif

} // namespace MR
