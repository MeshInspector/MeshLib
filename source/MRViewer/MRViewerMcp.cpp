#ifndef MESHLIB_NO_MCP

#include "MRMcp/MRMcp.h"
#include "MRMesh/MROnInit.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRUITestEngineControl.h"
#include "MRViewer/MRViewer.h"

namespace MR
{

static void skipFramesAfterInput()
{
    for ( int i = 0; i < MR::getViewerInstance().forceRedrawMinimumIncrementAfterEvents; ++i )
        MR::CommandLoop::runCommandFromGUIThread( [] {} ); // Wait a few frames.
}

static nlohmann::json mcpToolListUiEntries( const nlohmann::json& args )
{
    std::vector<UI::TestEngine::Control::TypedEntry> list;
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto ex = UI::TestEngine::Control::listEntries( args.at( "path" ).get<std::vector<std::string>>() );
        if ( !ex )
            throw std::runtime_error( ex.error() );
        list = std::move( *ex );
    } );

    nlohmann::json ret = nlohmann::json::array();
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

        ret.push_back( nlohmann::json::object( {
            { "name", elem.name },
            { "type", std::move( typeStr ) },
        } ) );
    }

    return nlohmann::json::object( { { "result", ret } } );
}

static nlohmann::json mcpToolPressButton( const nlohmann::json& args )
{
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto ex = UI::TestEngine::Control::pressButton( args.at( "path" ).get<std::vector<std::string>>() );
        if ( !ex )
            throw std::runtime_error( ex.error() );
    } );
    skipFramesAfterInput();

    return nlohmann::json::object();
}

template <typename T>
static nlohmann::json mcpToolReadValue( const nlohmann::json& args )
{
    UI::TestEngine::Control::Value<T> value;
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto ex = UI::TestEngine::Control::readValue<T>( args.at( "path" ).get<std::vector<std::string>>() );
        if ( !ex )
            throw std::runtime_error( ex.error() );
        value = std::move( *ex );
    } );

    nlohmann::json ret = nlohmann::json::object();
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

template <typename T>
static nlohmann::json mcpToolWriteValue( const nlohmann::json& args )
{
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto ex = UI::TestEngine::Control::writeValue<T>( args.at( "path" ).get<std::vector<std::string>>(), T( args.at( "value" ) ) );
        if ( !ex )
            throw std::runtime_error( ex.error() );
    } );
    skipFramesAfterInput();

    return nlohmann::json::object();
}

MR_ON_INIT{
    Mcp::Server& server = Mcp::getDefaultServer();

    server.addTool(
        /*id*/"ui.listEntries",
        /*name*/"List UI entries",
        /*desc*/"Returns the list of UI elements at the given path. The elements form a tree. Pass an empty array to get the top-level elements. Each element is described by a string. The path parameter describes the path from the root node to a specific element. Only elements of type `group` can have sub-elements.",
        /*input_schema*/Mcp::Schema::Object{}.addMember( "path", Mcp::Schema::Array( Mcp::Schema::String{} ) ),
        /*output_schema*/Mcp::Schema::Array( Mcp::Schema::Object{}.addMember( "name", Mcp::Schema::String{} ).addMember( "type", Mcp::Schema::String{} ) ),
        /*func*/mcpToolListUiEntries
    );

    server.addTool(
        /*id*/"ui.pressButton",
        /*name*/"Press button",
        /*desc*/"Presses the button at the given path.",
        /*input_schema*/Mcp::Schema::Object{}.addMember( "path", Mcp::Schema::Array( Mcp::Schema::String{} ) ),
        /*output_schema*/Mcp::Schema::Empty{},
        /*func*/mcpToolPressButton
    );

    auto handleValueType = [&]<typename T>( const std::string& typeName )
    {
        server.addTool(
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
            /*input_schema*/Mcp::Schema::Object{}.addMember( "path", Mcp::Schema::Array( Mcp::Schema::String{} ) ),
            /*output_schema*/(
                std::is_same_v<T, std::string>
                ?
                    static_cast<Mcp::Schema::Base &&>(
                        Mcp::Schema::Object{}.addMember( "value", Mcp::Schema::String{} ).addMemberOpt( "allowedValues", Mcp::Schema::Array( Mcp::Schema::String{} ) )
                    )
                :
                    static_cast<Mcp::Schema::Base &&>(
                        Mcp::Schema::Object{}.addMember( "value", Mcp::Schema::Number{} ).addMember( "min", Mcp::Schema::Number{} ).addMember( "max", Mcp::Schema::Number{} )
                    )
            ),
            /*func*/mcpToolReadValue<T>
        );

        server.addTool(
            /*id*/"ui.writeValue" + typeName,
            /*name*/"Write " + typeName + " value",
            /*desc*/"Writes the value at the given path, of type `" + typeName + "`. You can call `ui.readValue" + typeName + "` before this to know what values are allowed.",
            /*input_schema*/Mcp::Schema::Object{}.addMember( "path", Mcp::Schema::Array( Mcp::Schema::String{} ) ).addMember( "value", std::is_same_v<T, std::string> ? static_cast<Mcp::Schema::Base &&>( Mcp::Schema::String{} ) : static_cast<Mcp::Schema::Base &&>( Mcp::Schema::Number{} ) ),
            /*output_schema*/Mcp::Schema::Empty{},
            /*func*/mcpToolWriteValue<T>
        );
    };

    handleValueType.operator()<std::int64_t>( "Int" );
    handleValueType.operator()<std::uint64_t>( "Uint" );
    handleValueType.operator()<double>( "Real" );
    handleValueType.operator()<std::string>( "String" );
}; // MR_ON_INIT

} // namespace MR

#endif
