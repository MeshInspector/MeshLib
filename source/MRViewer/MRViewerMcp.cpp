#ifndef MESHLIB_NO_MCP

#include "MRMcp/MRMcp.h"
#include "MRMesh/MRBase64.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MROnInit.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector3.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRFitData.h"
#include "MRViewer/MRMcpCommon.h"
#include "MRViewer/MRMouse.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"

#include <GLFW/glfw3.h>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>

namespace MR::Mcp
{

namespace
{

Vector3f readVec3( const nlohmann::json& j, std::string_view field )
{
    if ( !j.is_array() || j.size() != 3 )
        throw std::runtime_error( fmt::format( "`{}` must be a 3-element array", field ) );
    const Vector3f v{ j[0].get<float>(), j[1].get<float>(), j[2].get<float>() };
    if ( !std::isfinite( v.x ) || !std::isfinite( v.y ) || !std::isfinite( v.z ) )
        throw std::runtime_error( fmt::format( "`{}` contains non-finite value", field ) );
    return v;
}

int parseModifiers( const nlohmann::json& args )
{
    int mods = 0;
    if ( !args.contains( "modifiers" ) )
        return mods;
    for ( const auto& j : args["modifiers"] )
    {
        const std::string m = j.get<std::string>();
        if ( m == "ctrl" )       mods |= GLFW_MOD_CONTROL;
        else if ( m == "shift" ) mods |= GLFW_MOD_SHIFT;
        else if ( m == "alt" )   mods |= GLFW_MOD_ALT;
        else if ( m == "super" ) mods |= GLFW_MOD_SUPER;
        else throw std::runtime_error( fmt::format( "Unknown modifier `{}` (expected `ctrl`/`shift`/`alt`/`super`)", m ) );
    }
    return mods;
}

MouseButton parseMouseButton( const std::string& b )
{
    if ( b == "left" )   return MouseButton::Left;
    if ( b == "right" )  return MouseButton::Right;
    if ( b == "middle" ) return MouseButton::Middle;
    throw std::runtime_error( fmt::format( "Unknown mouse button `{}` (expected `left`/`right`/`middle`)", b ) );
}

int parseKey( const std::string& s )
{
    // Single printable ASCII char — GLFW key codes are ASCII-uppercase for letters/digits/punctuation.
    if ( s.size() == 1 && std::isprint( static_cast<unsigned char>( s[0] ) ) )
        return std::toupper( static_cast<unsigned char>( s[0] ) );
    if ( s == "Escape" )    return GLFW_KEY_ESCAPE;
    if ( s == "Enter" || s == "Return" ) return GLFW_KEY_ENTER;
    if ( s == "Space" )     return GLFW_KEY_SPACE;
    if ( s == "Tab" )       return GLFW_KEY_TAB;
    if ( s == "Backspace" ) return GLFW_KEY_BACKSPACE;
    if ( s == "Delete" )    return GLFW_KEY_DELETE;
    if ( s == "Home" )      return GLFW_KEY_HOME;
    if ( s == "End" )       return GLFW_KEY_END;
    if ( s == "PageUp" )    return GLFW_KEY_PAGE_UP;
    if ( s == "PageDown" )  return GLFW_KEY_PAGE_DOWN;
    if ( s == "Left" || s == "ArrowLeft" )   return GLFW_KEY_LEFT;
    if ( s == "Right" || s == "ArrowRight" ) return GLFW_KEY_RIGHT;
    if ( s == "Up" || s == "ArrowUp" )       return GLFW_KEY_UP;
    if ( s == "Down" || s == "ArrowDown" )   return GLFW_KEY_DOWN;
    if ( s.size() >= 2 && s[0] == 'F' )
    {
        const int n = std::atoi( s.c_str() + 1 );
        if ( n >= 1 && n <= 25 ) return GLFW_KEY_F1 + ( n - 1 );
    }
    throw std::runtime_error( fmt::format( "Unknown key `{}` (use a single printable char or a name like `Escape`, `Enter`, `ArrowUp`, `F5`)", s ) );
}

} // namespace

static nlohmann::json mcpViewerFit( const nlohmann::json& args )
{
    const float factor = args.value( "factor", 1.0f );

    // Collect ids / points outside the GUI thread so parse errors throw before the main loop blocks.
    std::vector<uint64_t> ids;
    if ( args.contains( "objectIds" ) )
        for ( const auto& j : args["objectIds"] )
            ids.push_back( j.get<uint64_t>() );

    std::vector<Vector3f> points;
    if ( args.contains( "points" ) )
        for ( const auto& p : args["points"] )
            points.push_back( readVec3( p, "points[i]" ) );

    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto& vp = getViewerInstance().viewport();
        if ( ids.empty() && points.empty() )
        {
            FitDataParams p;
            p.factor = factor;
            vp.preciseFitDataToScreenBorder( p );
            return;
        }
        Box3f box;
        for ( uint64_t id : ids )
        {
            auto objEx = resolveId( id );
            if ( !objEx )
                throw std::runtime_error( objEx.error() );
            box.include( ( *objEx )->getWorldBox() );
        }
        for ( const Vector3f& pt : points )
            box.include( pt );
        if ( !box.valid() )
            throw std::runtime_error( "viewer.fit: the given objects and points contribute no bounding volume." );
        vp.preciseFitBoxToScreenBorder( { box, factor } );
    } );
    skipFramesAfterInput();
    return nlohmann::json::object();
}

static nlohmann::json mcpViewerSetupCamera( const nlohmann::json& args )
{
    const Vector3f forward = readVec3( args.at( "forwardDir" ), "forwardDir" );
    const Vector3f upRaw = readVec3( args.at( "upDir" ), "upDir" );

    if ( forward.lengthSq() < 1e-12f )
        throw std::runtime_error( "`forwardDir` must be non-zero." );
    const Vector3f fwdN = forward.normalized();

    // Orthogonalize upDir against forwardDir (Gram-Schmidt). cameraLookAlong asserts they are
    // perpendicular, so callers shouldn't have to compute it exactly.
    Vector3f up = upRaw - fwdN * dot( fwdN, upRaw );
    if ( up.lengthSq() < 1e-12f )
        throw std::runtime_error( "`upDir` is parallel to `forwardDir`; provide a non-parallel up vector." );
    up = up.normalized();

    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto& vp = getViewerInstance().viewport();
        vp.cameraLookAlong( fwdN, up );
        // preciseFitDataToScreenBorder defaults to snapView=false — keeps the user's direction.
        vp.preciseFitDataToScreenBorder();
    } );
    skipFramesAfterInput();
    return nlohmann::json::object();
}

static nlohmann::json mcpViewerCaptureScreenshot( const nlohmann::json& args )
{
    const bool includeUi = args.value( "includeUi", false );
    const int width = args.value( "width", 0 );
    const int height = args.value( "height", 0 );
    if ( width < 0 || height < 0 )
        throw std::runtime_error( "`width` and `height` must be non-negative." );
    const bool transparentBg = args.value( "transparentBg", false );

    Image img;
    if ( includeUi )
    {
        // captureUIScreenShot is async — it queues onto the GUI thread and fires the callback after
        // the next frame. Block the MCP handler thread on a future until that happens.
        std::promise<Image> promise;
        auto future = promise.get_future();
        getViewerInstance().captureUIScreenShot( [&promise]( const Image& i ) { promise.set_value( i ); } );
        img = future.get();
    }
    else
    {
        MR::CommandLoop::runCommandFromGUIThread( [&]
        {
            img = getViewerInstance().captureSceneScreenShot( { width, height }, transparentBg );
        } );
    }

    nlohmann::json out = nlohmann::json::object();
    if ( args.contains( "filePath" ) )
    {
        const auto path = pathFromUtf8( args["filePath"].get<std::string>() );
        auto saved = ImageSave::toAnySupportedFormat( img, path );
        if ( !saved )
            throw std::runtime_error( saved.error() );
        out["path"] = utf8string( path );
    }
    else
    {
        UniqueTemporaryFolder tmp{};
        const std::filesystem::path path = tmp / "screenshot.png";
        auto saved = ImageSave::toAnySupportedFormat( img, path );
        if ( !saved )
            throw std::runtime_error( saved.error() );
        std::ifstream in( path, std::ios::binary );
        if ( !in )
            throw std::runtime_error( fmt::format( "Could not read back temp file {}", utf8string( path ) ) );
        std::vector<std::uint8_t> bytes( ( std::istreambuf_iterator<char>( in ) ), std::istreambuf_iterator<char>() );
        out["bytes"] = encode64( bytes.data(), bytes.size() );
        out["contentType"] = "image/png";
    }
    out["width"] = img.resolution.x;
    out["height"] = img.resolution.y;
    return out;
}

static nlohmann::json mcpViewerSendMouseEvent( const nlohmann::json& args )
{
    const std::string type = args.at( "type" ).get<std::string>();
    const int mods = parseModifiers( args );
    auto& v = getViewerInstance();

    // mouseMove between hover-resolve and click matters for many plugins; follow the pattern used
    // by Python UI tests in test_ui/scenarios/scenario_helpers.py.
    const bool hasXY = args.contains( "x" ) && args.contains( "y" );
    auto doMove = [&]
    {
        const int x = args.at( "x" ).get<int>();
        const int y = args.at( "y" ).get<int>();
        v.emplaceEvent( "mcp mouseMove", [&v, x, y]{ v.mouseMove( x, y ); } );
        skipFramesAfterInput();
    };

    if ( type == "move" )
    {
        if ( !hasXY )
            throw std::runtime_error( "`move` requires `x` and `y`." );
        doMove();
    }
    else if ( type == "scroll" )
    {
        if ( !args.contains( "scrollDelta" ) )
            throw std::runtime_error( "`scroll` requires `scrollDelta` (positive = scroll up)." );
        const float delta = args["scrollDelta"].get<float>();
        v.emplaceEvent( "mcp mouseScroll", [&v, delta]{ v.mouseScroll( delta ); } );
        skipFramesAfterInput();
    }
    else if ( type == "down" || type == "up" || type == "click" )
    {
        if ( !args.contains( "button" ) )
            throw std::runtime_error( fmt::format( "`{}` requires `button` (`left`/`right`/`middle`).", type ) );
        const MouseButton btn = parseMouseButton( args["button"].get<std::string>() );
        if ( hasXY )
            doMove();
        if ( type == "down" || type == "click" )
        {
            v.emplaceEvent( "mcp mouseDown", [&v, btn, mods]{ v.mouseDown( btn, mods ); } );
            skipFramesAfterInput();
        }
        if ( type == "up" || type == "click" )
        {
            v.emplaceEvent( "mcp mouseUp", [&v, btn, mods]{ v.mouseUp( btn, mods ); } );
            skipFramesAfterInput();
        }
    }
    else
    {
        throw std::runtime_error( fmt::format( "Unknown `type` `{}` (expected `down`/`up`/`click`/`move`/`scroll`).", type ) );
    }
    return nlohmann::json::object();
}

static nlohmann::json mcpViewerSendKeyboardEvent( const nlohmann::json& args )
{
    const std::string type = args.at( "type" ).get<std::string>();
    const std::string keyStr = args.at( "key" ).get<std::string>();
    const int mods = parseModifiers( args );
    auto& v = getViewerInstance();

    if ( type == "press" )
    {
        // keyPressed takes a unicode codepoint (text-input semantics). Require a single-char string;
        // decoding full UTF-8 is out of scope for v1.
        if ( keyStr.size() != 1 || !std::isprint( static_cast<unsigned char>( keyStr[0] ) ) )
            throw std::runtime_error( "`press` requires a single printable ASCII character as `key` (for text input)." );
        const unsigned int unicode = static_cast<unsigned char>( keyStr[0] );
        v.emplaceEvent( "mcp keyPressed", [&v, unicode, mods]{ v.keyPressed( unicode, mods ); } );
    }
    else if ( type == "down" || type == "up" || type == "repeat" )
    {
        const int glfwKey = parseKey( keyStr );
        if ( type == "down" )
            v.emplaceEvent( "mcp keyDown", [&v, glfwKey, mods]{ v.keyDown( glfwKey, mods ); } );
        else if ( type == "up" )
            v.emplaceEvent( "mcp keyUp", [&v, glfwKey, mods]{ v.keyUp( glfwKey, mods ); } );
        else
            v.emplaceEvent( "mcp keyRepeat", [&v, glfwKey, mods]{ v.keyRepeat( glfwKey, mods ); } );
    }
    else
    {
        throw std::runtime_error( fmt::format( "Unknown `type` `{}` (expected `down`/`up`/`press`/`repeat`).", type ) );
    }
    skipFramesAfterInput();
    return nlohmann::json::object();
}

static nlohmann::json mcpViewerShutdown( const nlohmann::json& )
{
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        MR::getViewerInstance().stopEventLoop();
    } );
    return nlohmann::json::object();
}

MR_ON_INIT{
    Server& server = getDefaultServer();

    server.addTool(
        /*id*/  "viewer.fit",
        /*name*/"Fit camera to scene or a subset",
        /*desc*/"Frame a set of objects and/or world-space points in the active viewport. Pass `objectIds` (scene-object "
                "ids from `scene.listObjectTree`) and/or `points` (3-element `[x,y,z]` arrays). If both are present, "
                "their bounding volumes are unioned. If neither is given, fits the whole scene (same as the UI's "
                "\"Fit\" action). `factor` (default 1.0) controls framing margin — higher means more screen coverage. "
                "The camera angle is preserved (no canonical-view snapping).",
        /*input_schema*/Schema::Object{}
            .addMemberOpt( "objectIds", Schema::Array( Schema::Number{} ) )
            .addMemberOpt( "points",    Schema::Array( Schema::Array( Schema::Number{} ) ) )
            .addMemberOpt( "factor",    Schema::Number{} ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpViewerFit
    );

    server.addTool(
        /*id*/  "viewer.setupCamera",
        /*name*/"Set camera forward and up directions",
        /*desc*/"Point the camera along `forwardDir` (a world-space direction vector pointing from the camera toward "
                "the subject). `upDir` sets the screen-up direction; it is automatically orthogonalized against "
                "`forwardDir`, so it does not have to be exactly perpendicular, but it must not be parallel. The "
                "camera is always refit to the whole scene after reorientation.",
        /*input_schema*/Schema::Object{}
            .addMember( "forwardDir", Schema::Array( Schema::Number{} ) )
            .addMember( "upDir",      Schema::Array( Schema::Number{} ) ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpViewerSetupCamera
    );

    server.addTool(
        /*id*/  "viewer.captureScreenshot",
        /*name*/"Capture viewport screenshot",
        /*desc*/"Render the viewer to a PNG. Default (`includeUi: false`) captures only the 3D viewport; set "
                "`includeUi: true` to capture the whole window including panels, ribbon, and dialogs. "
                "For the 3D-only mode, optional `width`/`height` request a specific resolution (zero or missing = "
                "current viewport size) and `transparentBg` (default false) omits the background — these options "
                "are ignored when `includeUi` is true (window capture always uses the current framebuffer with its "
                "normal background). "
                "Pass `filePath` to save server-side; response is `{path, width, height}`. Omit `filePath` to get "
                "an inline base64 PNG; response is `{bytes, width, height}`.",
        /*input_schema*/Schema::Object{}
            .addMemberOpt( "includeUi",     Schema::Bool{} )
            .addMemberOpt( "width",         Schema::Number{} )
            .addMemberOpt( "height",        Schema::Number{} )
            .addMemberOpt( "transparentBg", Schema::Bool{} )
            .addMemberOpt( "filePath",      Schema::String{} ),
        /*output_schema*/Schema::Object{}
            .addMemberOpt( "path",   Schema::String{} )
            .addMemberOpt( "bytes",  Schema::String{} )
            .addMember(    "width",  Schema::Number{} )
            .addMember(    "height", Schema::Number{} ),
        /*func*/mcpViewerCaptureScreenshot
    );

    server.addTool(
        /*id*/  "viewer.sendMouseEvent",
        /*name*/"Inject a mouse event",
        /*desc*/"Queue a mouse event on the viewer's event loop. `type` is one of `down`, `up`, `click`, `move`, `scroll`. "
                "`button` is `left`/`right`/`middle` (required for `down`/`up`/`click`). `x`/`y` are window pixels, "
                "top-left origin (required for `move`; optional for `down`/`up`/`click` — if given, a mouseMove is "
                "sent first so hover state resolves before the button event). `scrollDelta` is required for `scroll` "
                "(positive = scroll up). `modifiers` is an array of any of `ctrl`/`shift`/`alt`/`super`. "
                "`click` is shorthand for `down` + `up` at the same position.",
        /*input_schema*/Schema::Object{}
            .addMember(    "type",        Schema::String{} )
            .addMemberOpt( "button",      Schema::String{} )
            .addMemberOpt( "x",           Schema::Number{} )
            .addMemberOpt( "y",           Schema::Number{} )
            .addMemberOpt( "scrollDelta", Schema::Number{} )
            .addMemberOpt( "modifiers",   Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpViewerSendMouseEvent
    );

    server.addTool(
        /*id*/  "viewer.sendKeyboardEvent",
        /*name*/"Inject a keyboard event",
        /*desc*/"Queue a keyboard event on the viewer's event loop. `type` is one of `down`, `up`, `press`, `repeat`. "
                "`key` is either a single printable character (e.g. `\"a\"`, `\"A\"`, `\"5\"`) or a named key "
                "(`Escape`, `Enter`, `Space`, `Tab`, `Backspace`, `Delete`, `Home`, `End`, `PageUp`, `PageDown`, "
                "`ArrowUp`/`ArrowDown`/`ArrowLeft`/`ArrowRight`, `F1`…`F25`). `press` is for text input — it emits a "
                "unicode character event and requires a single printable character. `down`/`up`/`repeat` send raw "
                "key events (use these for shortcuts like Ctrl+S, or for keys that don't produce text). `modifiers` "
                "is an array of any of `ctrl`/`shift`/`alt`/`super`.",
        /*input_schema*/Schema::Object{}
            .addMember(    "type",      Schema::String{} )
            .addMember(    "key",       Schema::String{} )
            .addMemberOpt( "modifiers", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpViewerSendKeyboardEvent
    );

    server.addTool(
        /*id*/  "viewer.shutdown",
        /*name*/"Close MeshInspector",
        /*desc*/"Cleanly stop MeshInspector's event loop and exit the process. Returns immediately so the MCP "
                "response can flush before the server socket closes; the actual shutdown happens on the next frame. "
                "After this call the gateway's `launch` tool can bring MeshInspector back up.",
        /*input_schema*/Schema::Object{},
        /*output_schema*/Schema::Object{},
        /*func*/mcpViewerShutdown
    );
}; // MR_ON_INIT

} // namespace MR::Mcp

#endif
