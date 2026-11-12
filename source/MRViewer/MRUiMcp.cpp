#ifndef MESHLIB_NO_MCP

#include "MRMcp/MRMcp.h"
#include "MRMesh/MROnInit.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRMcpCommon.h"
#include "MRViewer/MRProgressBar.h"
#include "MRViewer/MRUITestEngine.h"
#include "MRViewer/MRUITestEngineControl.h"

#include <algorithm>
#include <thread>

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
    runOnMainThread( [&]
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
    runOnMainThread( [&]
    {
        auto ex = UI::TestEngine::Control::listAllEntries( args.value( "path", std::vector<std::string>{} ) );
        if ( !ex )
            throw std::runtime_error( ex.error() );
        list = std::move( *ex );
    } );

    const auto filter = toLower( args.value( "filter", std::string{} ) );
    nlohmann::json ret = nlohmann::json::array();
    for ( const auto& pe : list )
    {
        if ( !filter.empty() && toLower( pe.second.name ).find( filter ) == std::string::npos )
            continue;
        ret.push_back( nlohmann::json::object( {
            { "path", pe.first },
            { "name", pe.second.name },
            { "type", mcpTypeStr( pe.second.type ) },
            { "status", pe.second.status },
        } ) );
    }

    return nlohmann::json::object( { { "result", ret } } );
}

// Drain any TestEngine status messages emitted during a just-dispatched UI action and surface
// them as a tool error. Call on the GUI thread (as `after` of `waitFramesAfterInput`) because TE state isn't thread-safe.
// Non-static so MRToolsMcp.cpp's `tools_action` handler can reuse it across TUs.
void surfaceTestEngineStatusMessages()
{
    const auto msgs = UI::TestEngine::consumeStatusMessages();
    if ( msgs.empty() )
        return;
    std::string joined = msgs.front();
    for ( size_t i = 1; i < msgs.size(); ++i )
        joined += '\n' + msgs[i];
    throw std::runtime_error( std::move( joined ) );
}

static nlohmann::json mcpToolPressButton( const nlohmann::json& args )
{
    const auto path = args.at( "path" ).get<std::vector<std::string>>();
    runOnMainThread( [&]
    {
        auto ex = UI::TestEngine::Control::pressButton( path );
        if ( !ex )
            throw std::runtime_error( ex.error() );
        // Non-empty = disabled status; surface to MCP as an error (empty = OK, click simulated).
        if ( !ex->empty() )
            throw std::runtime_error( fmt::format( "pressButton {}: {}", UI::TestEngine::Control::pathToString( path ), *ex ) );
    } );
    return inputResult( waitFramesAfterInput( surfaceTestEngineStatusMessages ) );
}

static nlohmann::json mcpToolStageFileDialogPaths( const nlohmann::json& args )
{
    std::vector<std::filesystem::path> paths;
    for ( const auto& p : args.at( "paths" ) )
        paths.push_back( pathFromUtf8( p.get<std::string>() ) );

    runOnMainThread( [&]
    {
        UI::TestEngine::stageFileDialogPaths( std::move( paths ) );
    } );
    return nlohmann::json::object();
}

template <typename T>
static nlohmann::json mcpToolReadValue( const nlohmann::json& args )
{
    UI::TestEngine::Control::Value<T> value;
    runOnMainThread( [&]
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
    const auto path = args.at( "path" ).get<std::vector<std::string>>();
    runOnMainThread( [&]
    {
        auto ex = UI::TestEngine::Control::writeValue<T>( path, T( args.at( "value" ) ) );
        if ( !ex )
            throw std::runtime_error( ex.error() );
        // Non-empty = disabled status; surface to MCP as an error (empty = OK, write simulated).
        if ( !ex->empty() )
            throw std::runtime_error( fmt::format( "writeValue {}: {}", UI::TestEngine::Control::pathToString( path ), *ex ) );
    } );
    return inputResult( waitFramesAfterInput( surfaceTestEngineStatusMessages ) );
}

// Prepended to every path-accepting tool's `desc`. One place defines the vocabulary;
// per-tool descs only say what the tool itself does.
static constexpr std::string_view kPathSemantics =
    "`path`: array of entry names from the root (empty = root). Whitespace-sensitive. "
    "Names may contain `##<suffix>` — a hidden ImGui uniqueness marker; pass the name VERBATIM "
    "as returned by `ui_listEntries` / `ui_listAllEntries`, do not strip `##<suffix>`.\n"
    "`type`: `button` | `group` | `int` | `uint` | `float` | `string`.\n"
    "`status`: `\"available\"` | `\"disabled: <reason>\"` (widget greyed out) | "
    "`\"disabled: blocked by modal '<name>'\"` (dismiss the modal first; a group says so when nothing in it accepts input).\n\n";

// Appended to the descriptions of the tools that send an input.
static constexpr std::string_view kBusySemantics =
    " Waits until the UI has processed the input, at most 20 seconds; if the action it started keeps MeshInspector "
    "busy longer, returns `busy` with an explanation instead of the empty result, see `ui_progressStatus`.";

static std::string withPreamble( std::string_view specific )
{
    return std::string( kPathSemantics ) + std::string( specific );
}

// Returns a snapshot of the global progress bar state, after waiting up to `waitSeconds` for MeshInspector to become idle.
// Read-only; safe to call any time.
static nlohmann::json mcpToolProgressStatus( const nlohmann::json& args )
{
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + ceil<milliseconds>( duration<double>(
        std::clamp( args.value( "waitSeconds", 0.0 ), 0.0, double( cMainThreadTimeout.count() ) ) ) );
    for ( ;; )
    {
        // a probe of at least 2 s: an action that runs in a frame without the progress bar blocks the main thread;
        // longer than the 1 s after which CommandLoop re-posts a lost wake-up event on Linux, not to report an idle thread as busy
        bool busy = false;
        try
        {
            CommandLoop::runCommandFromGUIThread( [] {}, std::max<milliseconds>( seconds( 2 ), ceil<milliseconds>( deadline - steady_clock::now() ) ) );
        }
        catch ( const CommandLoop::StartTimeout& )
        {
            busy = true;
        }
        // not `isFinished()`: the progress popup blocks `ui_*` until the post-processing, and operations ordered meanwhile follow it
        const bool active = MR::ProgressBar::isOrdered();
        if ( !busy && active && steady_clock::now() < deadline )
        {
            std::this_thread::sleep_for( milliseconds( 250 ) ); // the operation runs in a background thread
            continue;
        }
        nlohmann::json out = nlohmann::json::object( { { "busy", busy }, { "active", active } } );
        if ( busy )
            out["busyFor"] = duration_cast<seconds>( steady_clock::now() - CommandLoop::lastProcessTime() ).count();
        if ( active )
        {
            out["title"]   = MR::ProgressBar::getLastOperationTitle();
            out["percent"] = MR::ProgressBar::getProgress() * 100.f;
        }
        return out;
    }
}

MR_ON_INIT{
    Server& server = getDefaultServer();

    server.addTool(
        /*id*/"ui_listEntries",
        /*name*/"List UI entries (one level)",
        /*desc*/withPreamble( "List the direct children of `path`. Use `ui_listAllEntries` to get the entire subtree in one call." ),
        /*input_schema*/Schema::Object{}.addMember( "path", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Array(
            Schema::Object{}
                .addMember( "name", Schema::String{} )
                .addMember( "type", Schema::String{} )
                .addMember( "status", Schema::String{} )
        ),
        /*func*/mcpToolListUiEntries,
        /*readOnly*/true
    );

    server.addTool(
        /*id*/"ui_listAllEntries",
        /*name*/"List UI entries (full subtree)",
        /*desc*/withPreamble( "Flat depth-first dump of every entry in the subtree at `path` (omit or empty = whole tree). Each row carries its own `path`, so tree structure is recoverable. Prefer this for first-contact exploration; use `ui_listEntries` for a single level after a state change. "
            "The whole tree is large (over 10k characters): to find an entry, pass `filter`, which keeps only the entries whose name contains it, ignoring case." ),
        /*input_schema*/Schema::Object{}
            .addMemberOpt( "path", Schema::Array( Schema::String{} ) )
            .addMemberOpt( "filter", Schema::String{} ),
        /*output_schema*/Schema::Array(
            Schema::Object{}
                .addMember( "path", Schema::Array( Schema::String{} ) )
                .addMember( "name", Schema::String{} )
                .addMember( "type", Schema::String{} )
                .addMember( "status", Schema::String{} )
        ),
        /*func*/mcpToolListAllUiEntries,
        /*readOnly*/true
    );

    server.addTool(
        /*id*/"ui_pressButton",
        /*name*/"Click UI button",
        /*desc*/withPreamble( "Click the button at `path` (must end in a `type == \"button\"` entry). Fails if `status` starts with `\"disabled\"`." ) + std::string( kBusySemantics ),
        /*input_schema*/Schema::Object{}.addMember( "path", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Object{}.addMemberOpt( "busy", Schema::String{} ),
        /*func*/mcpToolPressButton
    );

    server.addTool(
        /*id*/"ui_stageFileDialogPaths",
        /*name*/"Stage paths for next TestEngine-triggered file dialog",
        /*desc*/
            "Pre-load the path(s) that the next OS file/folder dialog opened by a "
            "ui_pressButton-driven click will return. The dialog is skipped — no native "
            "modal appears — and the caller (Open / Save / Browse...) receives the staged "
            "paths. Single-shot: consumed by the next dialog. Use absolute paths in UTF-8. "
            "Stage before calling ui_pressButton on a button that opens a file dialog; if no "
            "paths are staged when such a button is pressed, ui_pressButton fails with a "
            "status message asking you to stage first. Validation: number of paths must "
            "match the dialog's multiselect mode; for open dialogs each path must exist with "
            "the right type (file vs directory); for save dialogs existence is not required "
            "but the path must not already exist as a directory. Real human clicks (not from "
            "ui_pressButton) always show the native dialog and ignore staged paths.",
        /*input_schema*/Schema::Object{}.addMember( "paths", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpToolStageFileDialogPaths
    );

    // (idSuffix, displayType) -- `idSuffix` is the CamelCase suffix on ui_readValue*/ui_writeValue*;
    // `displayType` matches the `type` field in listEntries output.
    auto handleValueType = [&]<typename T>( const std::string& idSuffix, const std::string& displayType )
    {
        const std::string readDesc = std::is_same_v<T, std::string>
            ? "Read the string value at `path`. If the response contains `allowedValues`, `ui_writeValue" + idSuffix + "` must pass one of those strings."
            : "Read the " + displayType + " value at `path`. Bounds `min`/`max` are inclusive for `ui_writeValue" + idSuffix + "`.";

        const std::string writeDesc = std::is_same_v<T, std::string>
            ? "Set the string value at `path`. Must match `allowedValues` (from `ui_readValue" + idSuffix + "`) when present. Fails if `status` starts with `\"disabled\"`."
            : "Set the " + displayType + " value at `path`. Must be within the inclusive `[min, max]` from `ui_readValue" + idSuffix + "`. Fails if `status` starts with `\"disabled\"`.";

        server.addTool(
            /*id*/"ui_readValue" + idSuffix,
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
            /*func*/mcpToolReadValue<T>,
            /*readOnly*/true
        );

        server.addTool(
            /*id*/"ui_writeValue" + idSuffix,
            /*name*/"Write UI " + displayType + " value",
            /*desc*/withPreamble( writeDesc ) + std::string( kBusySemantics ),
            /*input_schema*/Schema::Object{}.addMember( "path", Schema::Array( Schema::String{} ) ).addMember( "value", std::is_same_v<T, std::string> ? static_cast<Schema::Base &&>( Schema::String{} ) : static_cast<Schema::Base &&>( Schema::Number{} ) ),
            /*output_schema*/Schema::Object{}.addMemberOpt( "busy", Schema::String{} ),
            /*func*/mcpToolWriteValue<T>
        );
    };

    handleValueType.template operator()<std::int64_t >( "Int",    "int"    );
    handleValueType.template operator()<std::uint64_t>( "Uint",   "uint"   );
    handleValueType.template operator()<double       >( "Real",   "float"  );
    handleValueType.template operator()<std::string  >( "String", "string" );

    server.addTool(
        /*id*/"ui_progressStatus",
        /*name*/"Read progress bar state",
        /*desc*/"Snapshot of the global progress bar. Returns `{active: false}` if no long-running operation is in "
                "flight, else `{active: true, title: string, percent: number}`. `title` is the operation name passed "
                "to the underlying `ProgressBar::order(...)` (e.g. \"Boolean\", \"Loading\"). `percent` is 0-100. "
                "`busy` is true if the main thread did not respond within 2 seconds, e.g. while it runs an action "
                "without the progress bar, and `busyFor` then says for how many seconds it has not responded; other tools "
                "wait for it or report that MeshInspector is busy. Once `active` and `busy` are both false, the UI is "
                "responsive again. To wait for that, e.g. while `ui_*` dispatch is blocked by a `'Progress'` modal or after "
                "a tool returned `busy`, pass `waitSeconds` (at most 20) instead of polling: the call returns as soon as both "
                "are false, or after that many seconds with the state then.",
        /*input_schema*/Schema::Object{}.addMemberOpt( "waitSeconds", Schema::Number{} ),
        /*output_schema*/Schema::Object{}
            .addMember(    "busy",    Schema::Bool{} )
            .addMemberOpt( "busyFor", Schema::Number{} )
            .addMember(    "active",  Schema::Bool{} )
            .addMemberOpt( "title",   Schema::String{} )
            .addMemberOpt( "percent", Schema::Number{} ),
        /*func*/mcpToolProgressStatus,
        /*readOnly*/true
    );
}; // MR_ON_INIT

} // namespace MR::Mcp

#endif
