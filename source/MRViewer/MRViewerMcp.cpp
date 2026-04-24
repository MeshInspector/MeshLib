#ifndef MESHLIB_NO_MCP

#include "MRMcp/MRMcp.h"
#include "MRMesh/MROnInit.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRUITestEngineControl.h"

namespace MR::Mcp
{

// Hopefully "float" is more clear to LLMs than "real". The actual underlying type is `double`.
static const char* mcpTypeStr( UI::TestEngine::Control::EntryType type )
{
    switch ( type )
    {
        case UI::TestEngine::Control::EntryType::button:      return "button";
        case UI::TestEngine::Control::EntryType::group:       return "group";
        case UI::TestEngine::Control::EntryType::valueInt:    return "int";
        case UI::TestEngine::Control::EntryType::valueUint:   return "uint";
        case UI::TestEngine::Control::EntryType::valueReal:   return "float";
        case UI::TestEngine::Control::EntryType::valueString: return "string";
    }
    assert( false && "Unknown EntryType." );
    return "invalid";
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
        ret.push_back( nlohmann::json::object( {
            { "name", elem.name },
            { "type", mcpTypeStr( elem.type ) },
            { "status", elem.status },
        } ) );
    }

    return nlohmann::json::object( { { "result", ret } } );
}

static nlohmann::json mcpToolListAllUiEntries( const nlohmann::json& args )
{
    std::vector<UI::TestEngine::Control::PathedEntry> list;
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto ex = UI::TestEngine::Control::listAllEntries( args.value( "path", std::vector<std::string>{} ) );
        if ( !ex )
            throw std::runtime_error( ex.error() );
        list = std::move( *ex );
    } );

    nlohmann::json ret = nlohmann::json::array();
    for ( const auto& pe : list )
    {
        ret.push_back( nlohmann::json::object( {
            { "path", pe.first },
            { "name", pe.second.name },
            { "type", mcpTypeStr( pe.second.type ) },
            { "status", pe.second.status },
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

// Prepended to every path-accepting tool's `desc`. One place defines the vocabulary;
// per-tool descs only say what the tool itself does.
static constexpr std::string_view kPathSemantics =
    "`path`: array of entry names from the root (empty = root). Whitespace-sensitive. "
    "Names may contain `##<suffix>` — a hidden ImGui uniqueness marker; pass the name VERBATIM "
    "as returned by `ui.listEntries` / `ui.listAllEntries`, do not strip `##<suffix>`.\n"
    "`type`: `button` | `group` | `int` | `uint` | `float` | `string`.\n"
    "`status`: `\"available\"` | `\"disabled: <reason>\"` (widget greyed out) | "
    "`\"disabled: blocked by modal '<name>'\"` (dismiss the modal first).\n\n";

static std::string withPreamble( std::string_view specific )
{
    return std::string( kPathSemantics ) + std::string( specific );
}

MR_ON_INIT{
    Server& server = getDefaultServer();

    server.addTool(
        /*id*/"ui.listEntries",
        /*name*/"List UI entries (one level)",
        /*desc*/withPreamble( "List the direct children of `path`. Use `ui.listAllEntries` to get the entire subtree in one call." ),
        /*input_schema*/Schema::Object{}.addMember( "path", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Array(
            Schema::Object{}
                .addMember( "name", Schema::String{} )
                .addMember( "type", Schema::String{} )
                .addMember( "status", Schema::String{} )
        ),
        /*func*/mcpToolListUiEntries
    );

    server.addTool(
        /*id*/"ui.listAllEntries",
        /*name*/"List UI entries (full subtree)",
        /*desc*/withPreamble( "Flat depth-first dump of every entry in the subtree at `path` (omit or empty = whole tree). Each row carries its own `path`, so tree structure is recoverable. Prefer this for first-contact exploration; use `ui.listEntries` for a single level after a state change." ),
        /*input_schema*/Schema::Object{}.addMemberOpt( "path", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Array(
            Schema::Object{}
                .addMember( "path", Schema::Array( Schema::String{} ) )
                .addMember( "name", Schema::String{} )
                .addMember( "type", Schema::String{} )
                .addMember( "status", Schema::String{} )
        ),
        /*func*/mcpToolListAllUiEntries
    );

    server.addTool(
        /*id*/"ui.pressButton",
        /*name*/"Click UI button",
        /*desc*/withPreamble( "Click the button at `path` (must end in a `type == \"button\"` entry). Fails if `status` starts with `\"disabled\"`." ),
        /*input_schema*/Schema::Object{}.addMember( "path", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Empty{},
        /*func*/mcpToolPressButton
    );

    // (idSuffix, displayType) -- `idSuffix` is the CamelCase suffix on ui.readValue*/ui.writeValue*;
    // `displayType` matches the `type` field in listEntries output.
    auto handleValueType = [&]<typename T>( const std::string& idSuffix, const std::string& displayType )
    {
        const std::string readDesc = std::is_same_v<T, std::string>
            ? "Read the string value at `path`. If the response contains `allowedValues`, `ui.writeValue" + idSuffix + "` must pass one of those strings."
            : "Read the " + displayType + " value at `path`. Bounds `min`/`max` are inclusive for `ui.writeValue" + idSuffix + "`.";

        const std::string writeDesc = std::is_same_v<T, std::string>
            ? "Set the string value at `path`. Must match `allowedValues` (from `ui.readValue" + idSuffix + "`) when present. Fails if `status` starts with `\"disabled\"`."
            : "Set the " + displayType + " value at `path`. Must be within the inclusive `[min, max]` from `ui.readValue" + idSuffix + "`. Fails if `status` starts with `\"disabled\"`.";

        server.addTool(
            /*id*/"ui.readValue" + idSuffix,
            /*name*/"Read UI " + displayType + " value",
            /*desc*/withPreamble( readDesc ),
            /*input_schema*/Schema::Object{}.addMember( "path", Schema::Array( Schema::String{} ) ),
            /*output_schema*/(
                std::is_same_v<T, std::string>
                ?
                    static_cast<Schema::Base &&>(
                        Schema::Object{}.addMember( "value", Schema::String{} ).addMemberOpt( "allowedValues", Schema::Array( Schema::String{} ) )
                    )
                :
                    static_cast<Schema::Base &&>(
                        Schema::Object{}.addMember( "value", Schema::Number{} ).addMember( "min", Schema::Number{} ).addMember( "max", Schema::Number{} )
                    )
            ),
            /*func*/mcpToolReadValue<T>
        );

        server.addTool(
            /*id*/"ui.writeValue" + idSuffix,
            /*name*/"Write UI " + displayType + " value",
            /*desc*/withPreamble( writeDesc ),
            /*input_schema*/Schema::Object{}.addMember( "path", Schema::Array( Schema::String{} ) ).addMember( "value", std::is_same_v<T, std::string> ? static_cast<Schema::Base &&>( Schema::String{} ) : static_cast<Schema::Base &&>( Schema::Number{} ) ),
            /*output_schema*/Schema::Empty{},
            /*func*/mcpToolWriteValue<T>
        );
    };

    handleValueType.operator()<std::int64_t >( "Int",    "int"    );
    handleValueType.operator()<std::uint64_t>( "Uint",   "uint"   );
    handleValueType.operator()<double       >( "Real",   "float"  );
    handleValueType.operator()<std::string  >( "String", "string" );
}; // MR_ON_INIT

} // namespace MR::Mcp

#endif
