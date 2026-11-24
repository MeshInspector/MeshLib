#ifndef MESHLIB_NO_MCP

#include "MRMcp/MRMcp.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBase64.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRFinally.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRImageLoad.h"
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
#include "MRViewer/MRShortcutManager.h"
#include "MRViewer/MRUITestEngineControl.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewerSignals.h"
#include "MRViewer/MRViewport.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <stdexcept>
#include <thread>

namespace MR::Mcp
{

namespace
{

template <typename V>
V readVec( const nlohmann::json& j, std::string_view field )
{
    if ( !j.is_array() || j.size() != V::elements )
        throw std::runtime_error( fmt::format( "`{}` must be a {}-element array", field, V::elements ) );
    V v;
    for ( int i = 0; i < V::elements; ++i )
        if ( !std::isfinite( v[i] = j[i].get<float>() ) )
            throw std::runtime_error( fmt::format( "`{}` contains non-finite value", field ) );
    return v;
}

float readAnimationSeconds( const nlohmann::json& args )
{
    const float seconds = args.value( "animationSeconds", 0.f );
    if ( !( seconds >= 0 && seconds <= 10 ) )
        throw std::runtime_error( "`animationSeconds` must be in [0, 10]." );
    return seconds;
}

// waits until the camera of the viewport is not animated: the animation has ended, was stopped or was never started (no GL);
// at most seconds + 1 s, in case others keep starting new animations
void waitCameraAnimation( ViewportId vid, float seconds )
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds( int( ( seconds + 1 ) * 1000 ) );
    for ( ;; )
    {
        bool animated = false;
        runOnMainThread( [&] { animated = getViewerInstance().viewport( vid ).getCameraAnimationTarget() != nullptr; } );
        if ( !animated || std::chrono::steady_clock::now() > deadline )
            return;
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    }
}

int parseModifiers( const nlohmann::json& args )
{
    int mods = 0;
    if ( !args.contains( "modifiers" ) )
        return mods;
    for ( const auto& j : args["modifiers"] )
    {
        const std::string m = j.get<std::string>();
        if ( auto mod = ShortcutManager::parseModifier( m ) )
            mods |= *mod;
        else
            throw std::runtime_error( fmt::format( "Unknown modifier `{}` (expected `ctrl`/`shift`/`alt`/`super`/`primary`/`secondary`)", m ) );
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
    if ( auto key = ShortcutManager::parseKey( s ) )
        return *key;
    if ( auto mod = ShortcutManager::parseModifier( s ) ) // the left key of a modifier
        return *mod == GLFW_MOD_CONTROL ? GLFW_KEY_LEFT_CONTROL : *mod == GLFW_MOD_SHIFT ? GLFW_KEY_LEFT_SHIFT
            : *mod == GLFW_MOD_ALT ? GLFW_KEY_LEFT_ALT : GLFW_KEY_LEFT_SUPER;
    throw std::runtime_error( fmt::format( "Unknown key `{}` (use a single printable char or a name like `Escape`, `Enter`, `ArrowUp`, `F5`, `Num7`, `Delete`, `ForwardDelete`, `Ctrl`, `Shift`)", s ) );
}

// modifier keys (GLFW_MOD_* bits) held down by `viewer_sendKeyboardEvent`; main thread only
int gKeyboardMods = 0;

// makes ImGui see `mods` held in addition to the real keyboard, since it reads the real keyboard, not the injected key events;
// only writes the test-engine value, whose read-back lags by up to two frames; call on the main thread
void setImGuiModifiers( int mods )
{
    (void)UI::TestEngine::Control::writeValue<std::int64_t>( { "##key_modifiers" }, mods ); // ImGuiMenu::testKeyModifiers_
}

// the UI gets mouse and keyboard input, and screenshots are rendered, only with OpenGL, which -noWindow mode lacks
void requireOpenGL( const char* toolId )
{
    if ( !getViewerInstance().isGLInitialized() )
        throw std::runtime_error( fmt::format( "`{}` needs a window with OpenGL, it is not available in -noWindow mode.", toolId ) );
}

} // namespace

static nlohmann::json mcpViewerFit( const nlohmann::json& args )
{
    const float factor = args.value( "factor", 1.0f );
    const float seconds = readAnimationSeconds( args );

    // Collect ids / points outside the GUI thread so parse errors throw before the main loop blocks.
    std::vector<uint64_t> ids;
    if ( args.contains( "objectIds" ) )
        for ( const auto& j : args["objectIds"] )
            ids.push_back( j.get<uint64_t>() );

    std::vector<Vector3f> points;
    if ( args.contains( "points" ) )
        for ( const auto& p : args["points"] )
            points.push_back( readVec<Vector3f>( p, "points[i]" ) );

    ViewportId vid;
    runOnMainThread( [&]
    {
        auto& vp = getViewerInstance().viewport();
        vid = vp.id;
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
        if ( ( !ids.empty() || !points.empty() ) && !box.valid() )
            throw std::runtime_error( "viewer_fit: the given objects and points contribute no bounding volume." );
        // preciseFit...ToScreenBorder, animated unless seconds = 0; during an animation, from its target
        const auto camera = vp.getCameraAnimationTarget() ? *vp.getCameraAnimationTarget() : vp.getParameters();
        Box3f sceneBox;
        vp.animateCamera( box.valid() ? vp.calcPreciseFitBoxParameters( camera, { box, factor }, &sceneBox )
            : vp.calcPreciseFitDataParameters( camera, { factor }, &sceneBox ), seconds, sceneBox );
    } );
    waitCameraAnimation( vid, seconds );
    return nlohmann::json::object();
}

static nlohmann::json mcpViewerSetupCamera( const nlohmann::json& args )
{
    const Vector3f forward = readVec<Vector3f>( args.at( "forwardDir" ), "forwardDir" );
    const Vector3f upRaw = readVec<Vector3f>( args.at( "upDir" ), "upDir" );

    if ( forward.lengthSq() < 1e-12f )
        throw std::runtime_error( "`forwardDir` must be non-zero." );
    const Vector3f fwdN = forward.normalized();

    // Orthogonalize upDir against forwardDir (Gram-Schmidt). cameraLookAlong asserts they are
    // perpendicular, so callers shouldn't have to compute it exactly.
    Vector3f up = upRaw - fwdN * dot( fwdN, upRaw );
    if ( up.lengthSq() < 1e-12f )
        throw std::runtime_error( "`upDir` is parallel to `forwardDir`; provide a non-parallel up vector." );
    up = up.normalized();

    const std::string fit = args.value( "fit", std::string( "visible" ) );
    std::optional<FitMode> fitMode;
    if ( fit == "visible" )
        fitMode = FitMode::Visible;
    else if ( fit == "selected" )
        fitMode = FitMode::SelectedObjects;
    else if ( fit != "none" )
        throw std::runtime_error( fmt::format( "Unknown `fit` `{}` (expected `visible`/`selected`/`none`).", fit ) );
    const float factor = args.value( "factor", 1.0f );
    const float seconds = readAnimationSeconds( args );

    ViewportId vid;
    runOnMainThread( [&]
    {
        auto& vp = getViewerInstance().viewport();
        vid = vp.id;
        // cameraLookAlong and preciseFitDataToScreenBorder, animated unless seconds = 0; during an animation, from its target
        auto camera = vp.getCameraAnimationTarget() ? *vp.getCameraAnimationTarget() : vp.getParameters();
        camera.cameraTrackballAngle = Quaternionf( lookAt( fwdN, Vector3f{}, up ).A );
        Box3f sceneBox;
        // snapView=false keeps the user's direction.
        if ( fitMode )
            camera = vp.calcPreciseFitDataParameters( camera, { factor, false, *fitMode }, &sceneBox );
        vp.animateCamera( camera, seconds, sceneBox );
    } );
    waitCameraAnimation( vid, seconds );
    return nlohmann::json::object();
}

static nlohmann::json mcpViewerSetBackground( const nlohmann::json& args )
{
    std::optional<Color> color;
    if ( args.contains( "color" ) )
        color = parseColor( args["color"], "color" );

    // the image is loaded outside the GUI thread; empty optional removes it
    std::optional<std::optional<Viewport::BackgroundTexture>> texture;
    if ( const auto path = args.value( "imagePath", std::string() ); !path.empty() )
    {
        auto& bg = texture.emplace().emplace();
        const auto mode = args.value( "stretchMode", std::string( "fill" ) );
        constexpr std::array<std::string_view, 4> modes{ "none", "stretch", "fit", "fill" }; // StretchMode order
        const auto it = std::find( modes.begin(), modes.end(), mode );
        if ( it == modes.end() )
            throw std::runtime_error( fmt::format( "Unknown `stretchMode` `{}` (expected `none`/`stretch`/`fit`/`fill`).", mode ) );
        bg.stretchMode = Viewport::BackgroundTexture::StretchMode( it - modes.begin() );
        if ( args.contains( "startPos" ) )
            bg.startPos = readVec<Vector2f>( args["startPos"], "startPos" );
        if ( args.contains( "scaling" ) )
            bg.scaling = readVec<Vector2f>( args["scaling"], "scaling" );
        if ( !( bg.scaling.x > 0 && bg.scaling.y > 0 ) )
            throw std::runtime_error( "`scaling` must be positive." );
        auto image = ImageLoad::fromAnySupportedFormat( pathFromUtf8( path ) );
        if ( !image )
            throw std::runtime_error( image.error() );
        static_cast<Image&>( bg.texture ) = std::move( *image );
        bg.texture.filter = FilterType::Linear;
    }
    else if ( args.contains( "stretchMode" ) || args.contains( "startPos" ) || args.contains( "scaling" ) )
        throw std::runtime_error( "`stretchMode`, `startPos` and `scaling` need a non-empty `imagePath`." );
    else if ( args.contains( "imagePath" ) )
        texture.emplace();

    runOnMainThread( [&]
    {
        auto& viewer = getViewerInstance();
        auto rest = viewer.getPresentViewports();
        for ( ViewportId vid : viewer.getPresentViewports() )
        {
            rest.set( vid, false );
            if ( color )
                viewer.viewport( vid ).setBackgroundColor( *color );
            if ( texture ) // the image is moved into the last viewport, not copied
                viewer.viewport( vid ).setBackgroundTexture( rest.empty() ? std::move( *texture ) : *texture );
        }
    } );
    return nlohmann::json::object();
}

static nlohmann::json mcpViewerGetCamera( const nlohmann::json& )
{
    nlohmann::json out;
    runOnMainThread( [&]
    {
        const auto& vp = getViewerInstance().viewport();
        const auto vec = []( const Vector3f& v ) { return nlohmann::json::array( { v.x, v.y, v.z } ); };
        const auto& params = vp.getParameters();
        out = nlohmann::json::object( {
            { "forwardDir",   vec( -vp.getBackwardDirection() ) },
            { "upDir",        vec( vp.getUpDirection() ) },
            { "position",     vec( vp.getCameraPoint() ) },
            { "target",       vec( -params.cameraTranslation ) },
            { "orthographic", params.orthographic },
            { "zoom",         params.cameraZoom },
            { "viewAngle",    params.cameraViewAngle },
        } );
    } );
    return out;
}

static nlohmann::json mcpViewerCaptureScreenshot( const nlohmann::json& args )
{
    requireOpenGL( "viewer_captureScreenshot" );
    const bool includeUi = args.value( "includeUi", true );
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
        auto promise = std::make_shared<std::promise<Image>>(); // outlives this call if the main thread is busy
        auto future = promise->get_future();
        getViewerInstance().captureUIScreenShot( [promise]( const Image& i ) { promise->set_value( i ); } );
        if ( future.wait_for( cMainThreadTimeout ) != std::future_status::ready )
            throw busyError( fmt::format( "The main thread is busy: it did not capture the screenshot within {} seconds", cMainThreadTimeout.count() ) );
        img = future.get();
    }
    else
    {
        runOnMainThread( [&]
        {
            img = getViewerInstance().captureSceneScreenShot( { width, height }, transparentBg );
        } );
    }

    nlohmann::json structured = nlohmann::json::object();
    structured["width"]  = img.resolution.x;
    structured["height"] = img.resolution.y;

    if ( args.contains( "filePath" ) )
    {
        const auto path = pathFromUtf8( args["filePath"].get<std::string>() );
        auto saved = ImageSave::toAnySupportedFormat( img, path );
        if ( !saved )
            throw std::runtime_error( saved.error() );
        structured["path"] = utf8string( path );
        return structured;
    }

    UniqueTemporaryFolder tmp{};
    const std::filesystem::path path = tmp / "screenshot.png";
    auto saved = ImageSave::toAnySupportedFormat( img, path );
    if ( !saved )
        throw std::runtime_error( saved.error() );
    std::ifstream in( path, std::ios::binary );
    if ( !in )
        throw std::runtime_error( fmt::format( "Could not read back temp file {}", utf8string( path ) ) );
    std::vector<std::uint8_t> bytes( ( std::istreambuf_iterator<char>( in ) ), std::istreambuf_iterator<char>() );

    // MCP image content block + structured {width, height}.
    return nlohmann::json{
        { "content", nlohmann::json::array( {
            {
                { "type", "image" },
                { "data", encode64( bytes.data(), bytes.size() ) },
                { "mimeType", "image/png" },
            },
        } ) },
        { "structuredContent", structured },
    };
}

static nlohmann::json mcpViewerSendMouseEvent( const nlohmann::json& args )
{
    requireOpenGL( "viewer_sendMouseEvent" );
    const std::string type = args.at( "type" ).get<std::string>();
    const int mods = parseModifiers( args );
    auto& v = getViewerInstance();

    // mouseMove between hover-resolve and click matters for many plugins; follow the pattern used
    // by Python UI tests in test_ui/scenarios/scenario_helpers.py.
    const bool hasXY = args.contains( "x" ) && args.contains( "y" );
    // ImGui widgets see the modifiers during this call's events too, e.g. Ctrl+click in the scene list
    if ( mods )
        v.emplaceEvent( "mcp hold modifiers", [mods] { setImGuiModifiers( gKeyboardMods | mods ); } );
    MR_FINALLY
    {
        if ( mods )
            v.emplaceEvent( "mcp release modifiers", [] { setImGuiModifiers( gKeyboardMods ); } );
    };
    // the events are queued in order even if the main thread gets busy, then only waiting for it stops
    std::string busy;
    auto waitFrames = [&busy]
    {
        if ( busy.empty() )
            busy = waitFramesAfterInput();
    };
    auto doMove = [&]
    {
        const int x = args.at( "x" ).get<int>();
        const int y = args.at( "y" ).get<int>();
        v.emplaceEvent( "mcp mouseMove", [&v, x, y]{ v.mouseMove( x, y ); } );
        waitFrames();
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
        waitFrames();
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
            v.emplaceEvent( "mcp mouseDown", [&v, btn, mods]{ v.mouseDown( btn, mods | gKeyboardMods ); } );
            waitFrames();
        }
        if ( type == "up" || type == "click" )
        {
            v.emplaceEvent( "mcp mouseUp", [&v, btn, mods]{ v.mouseUp( btn, mods | gKeyboardMods ); } );
            waitFrames();
        }
    }
    else
    {
        throw std::runtime_error( fmt::format( "Unknown `type` `{}` (expected `down`/`up`/`click`/`move`/`scroll`).", type ) );
    }
    return inputResult( std::move( busy ) );
}

static nlohmann::json mcpViewerSendKeyboardEvent( const nlohmann::json& args )
{
    requireOpenGL( "viewer_sendKeyboardEvent" );
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
        v.emplaceEvent( "mcp keyPressed", [&v, unicode, mods]{ v.keyPressed( unicode, mods | gKeyboardMods ); } );
    }
    else if ( type == "down" || type == "up" || type == "repeat" )
    {
        const int glfwKey = parseKey( keyStr );
        // a modifier key stays held, also for ImGui widgets and later mouse and key events, until its `up`
        if ( const auto mod = ShortcutManager::parseModifier( keyStr ); mod && type != "repeat" )
            v.emplaceEvent( "mcp key modifiers", [mod = *mod, down = type == "down"]
                { setImGuiModifiers( gKeyboardMods = down ? gKeyboardMods | mod : gKeyboardMods & ~mod ); } );
        if ( type == "down" )
            v.emplaceEvent( "mcp keyDown", [&v, glfwKey, mods]{ v.keyDown( glfwKey, mods | gKeyboardMods ); } );
        else if ( type == "up" )
            v.emplaceEvent( "mcp keyUp", [&v, glfwKey, mods]{ v.keyUp( glfwKey, mods | gKeyboardMods ); } );
        else
            v.emplaceEvent( "mcp keyRepeat", [&v, glfwKey, mods]{ v.keyRepeat( glfwKey, mods | gKeyboardMods ); } );
    }
    else
    {
        throw std::runtime_error( fmt::format( "Unknown `type` `{}` (expected `down`/`up`/`press`/`repeat`).", type ) );
    }
    return inputResult( waitFramesAfterInput() );
}

static nlohmann::json mcpViewerSendDragDropEvent( const nlohmann::json& args )
{
    requireOpenGL( "viewer_sendDragDropEvent" );
    const std::string type = args.at( "type" ).get<std::string>();
    const bool hasXY = args.contains( "x" ) && args.contains( "y" );
    if ( type != "enter" && type != "over" && type != "leave" && type != "drop" )
        throw std::runtime_error( fmt::format( "Unknown `type` `{}` (expected `enter`/`over`/`leave`/`drop`).", type ) );
    if ( type == "over" && !hasXY )
        throw std::runtime_error( "`over` requires `x` and `y`." );
    std::vector<std::filesystem::path> paths;
    if ( type == "drop" )
        for ( const auto& p : args.at( "paths" ) )
            paths.push_back( pathFromUtf8( p.get<std::string>() ) );

    auto& v = getViewerInstance();
    // an entrance resets the drag position, so `enter` sends it before the `over`
    auto entrance = [&] { v.emplaceEvent( "mcp drag entrance", [&v, entered = type == "enter"] { v.signals().dragEntranceSignal( entered ); } ); };
    if ( type == "enter" )
        entrance();
    if ( hasXY ) // the drop lands at the last drag-over position, see MouseController::preDragDrop_
        v.emplaceEvent( "mcp drag over", [&v, x = args["x"].get<int>(), y = args["y"].get<int>()] { v.signals().dragOverSignal( x, y ); } );
    if ( type == "drop" )
        v.emplaceEvent( "mcp drop", [&v, paths = std::move( paths )] { v.dragDrop( paths ); } );
    else if ( type == "leave" )
        entrance();
    return inputResult( waitFramesAfterInput() );
}

static nlohmann::json mcpViewerSendTouchpadEvent( const nlohmann::json& args )
{
    requireOpenGL( "viewer_sendTouchpadEvent" );
    const std::string gesture = args.at( "gesture" ).get<std::string>();
    const std::string phase = args.at( "phase" ).get<std::string>();
    const int i = phase == "begin" ? 0 : phase == "update" ? 1 : phase == "end" ? 2 : -1;
    if ( i < 0 )
        throw std::runtime_error( fmt::format( "Unknown `phase` `{}` (expected `begin`/`update`/`end`).", phase ) );
    const float angle = args.value( "angle", 0.f ), dx = args.value( "dx", 0.f ), dy = args.value( "dy", 0.f ), scale = args.value( "scale", 1.f );
    const bool kinetic = args.value( "kinetic", false );

    auto& v = getViewerInstance();
    std::array<ViewerEventCallback, 3> events;
    if ( gesture == "rotate" )
        events = { [&v] { v.touchpadRotateGestureBegin(); }, [&v, angle] { v.touchpadRotateGestureUpdate( angle ); }, [&v] { v.touchpadRotateGestureEnd(); } };
    else if ( gesture == "swipe" )
        events = { [&v] { v.touchpadSwipeGestureBegin(); }, [&v, dx, dy, kinetic] { v.touchpadSwipeGestureUpdate( dx, dy, kinetic ); }, [&v] { v.touchpadSwipeGestureEnd(); } };
    else if ( gesture == "zoom" )
        events = { [&v] { v.touchpadZoomGestureBegin(); }, [&v, scale, kinetic] { v.touchpadZoomGestureUpdate( scale, kinetic ); }, [&v] { v.touchpadZoomGestureEnd(); } };
    else
        throw std::runtime_error( fmt::format( "Unknown `gesture` `{}` (expected `rotate`/`swipe`/`zoom`).", gesture ) );
    v.emplaceEvent( "mcp touchpad " + gesture + " " + phase, std::move( events[i] ) );
    return inputResult( waitFramesAfterInput() );
}

static nlohmann::json mcpViewerShutdown( const nlohmann::json& args )
{
    const bool force = args.value( "force", false );
    runOnMainThread( [&]
    {
        MR::getViewerInstance().stopEventLoop( 0, !force );
    } );
    return nlohmann::json::object();
}

MR_ON_INIT{
    Server& server = getDefaultServer();

    server.addTool(
        /*id*/  "viewer_fit",
        /*name*/"Fit camera to scene or a subset",
        /*desc*/"Frame a set of objects and/or world-space points in the active viewport. Pass `objectIds` (scene-object "
                "ids from `scene_listObjectTree`) and/or `points` (3-element `[x,y,z]` arrays). If both are present, "
                "their bounding volumes are unioned. If neither is given, fits the whole scene (same as the UI's "
                "\"Fit\" action). `factor` (default 1.0) controls framing margin — higher means more screen coverage. "
                "The camera angle is preserved (no canonical-view snapping). "
                "`animationSeconds` (default 0, at most 10) moves the camera smoothly to the new view during that time; "
                "the call returns when it has arrived.",
        /*input_schema*/Schema::Object{}
            .addMemberOpt( "objectIds", Schema::Array( Schema::Number{} ) )
            .addMemberOpt( "points",    Schema::Array( Schema::Base( Schema::Array( Schema::Number{} ) ) ) )
            .addMemberOpt( "factor",    Schema::Number{} )
            .addMemberOpt( "animationSeconds", Schema::Number{} ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpViewerFit
    );

    server.addTool(
        /*id*/  "viewer_setupCamera",
        /*name*/"Set camera forward and up directions",
        /*desc*/"Point the camera along `forwardDir` (a world-space direction vector pointing from the camera toward "
                "the subject). `upDir` sets the screen-up direction; it is automatically orthogonalized against "
                "`forwardDir`, so it does not have to be exactly perpendicular, but it must not be parallel. "
                "`fit` (default `visible`) refits the camera after reorientation: `visible` fits the geometry of all "
                "visible objects, `selected` fits the selected objects, `none` keeps the zoom and the camera's distance "
                "to the point it looks at, `target` from `viewer_getCamera` (the camera turns around that point, so "
                "its position still changes). "
                "The fit uses the geometry itself, so it frames tighter than `viewer_fit` with `objectIds`, which "
                "fits the objects' bounding boxes. `factor` (default 1.0) is the same framing margin as in `viewer_fit`. "
                "`animationSeconds` (default 0, at most 10) turns and moves the camera smoothly to the new view during that time; "
                "the call returns when it has arrived.",
        /*input_schema*/Schema::Object{}
            .addMember(    "forwardDir", Schema::Array( Schema::Number{} ) )
            .addMember(    "upDir",      Schema::Array( Schema::Number{} ) )
            .addMemberOpt( "fit",        Schema::String( std::vector<std::string>{ "visible", "selected", "none" } ) )
            .addMemberOpt( "factor",     Schema::Number{} )
            .addMemberOpt( "animationSeconds", Schema::Number{} ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpViewerSetupCamera
    );

    server.addTool(
        /*id*/  "viewer_setBackground",
        /*name*/"Set viewport background",
        /*desc*/"Change the background of all viewports: the color, the image or both; what is not given stays. `color` is "
                "`[r, g, b]` or `[r, g, b, a]` floats in [0, 1]. `imagePath` loads an image file (e.g. PNG or JPEG) drawn over "
                "the color and below the scene; an empty string removes the image. Only with a non-empty `imagePath`, "
                "`stretchMode` (default `fill`) is `fill` (covers the viewport, centered and cropped), `fit` (the whole image in "
                "the center, the color around it), `stretch` (covers the viewport exactly, the aspect ratio is not kept) or "
                "`none`: then `startPos` `[x, y]` (default `[0, 0]`) places the image's upper-left corner in framebuffer pixels "
                "from the viewport's upper-left corner, Y down, and `scaling` `[sx, sy]` (default `[1, 1]`, positive) is the size "
                "of an image pixel in framebuffer pixels. The image is not saved anywhere.",
        /*input_schema*/Schema::Object{}
            .addMemberOpt( "color",       Schema::Array( Schema::Number{} ) )
            .addMemberOpt( "imagePath",   Schema::String{} )
            .addMemberOpt( "stretchMode", Schema::String( std::vector<std::string>{ "fill", "fit", "stretch", "none" } ) )
            .addMemberOpt( "startPos",    Schema::Array( Schema::Number{} ) )
            .addMemberOpt( "scaling",     Schema::Array( Schema::Number{} ) ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpViewerSetBackground
    );

    server.addTool(
        /*id*/  "viewer_getCamera",
        /*name*/"Get camera state",
        /*desc*/"Read the active viewport's camera. `forwardDir` and `upDir` are unit world-space vectors in the same "
                "convention as `viewer_setupCamera` accepts, so passing them back with `fit: none` restores the "
                "orientation. `position` is the camera point in world space; `target` is the world point at the center "
                "of the view, which the camera turns around in `viewer_setupCamera` with `fit: none` (keeping its distance "
                "to it). `orthographic` is the projection mode, "
                "`zoom` is the camera zoom factor and `viewAngle` the vertical field of view in degrees.",
        /*input_schema*/Schema::Object{},
        /*output_schema*/Schema::Object{}
            .addMember( "forwardDir",   Schema::Array( Schema::Number{} ) )
            .addMember( "upDir",        Schema::Array( Schema::Number{} ) )
            .addMember( "position",     Schema::Array( Schema::Number{} ) )
            .addMember( "target",       Schema::Array( Schema::Number{} ) )
            .addMember( "orthographic", Schema::Bool{} )
            .addMember( "zoom",         Schema::Number{} )
            .addMember( "viewAngle",    Schema::Number{} ),
        /*func*/mcpViewerGetCamera,
        /*readOnly*/true
    );

    server.addTool(
        /*id*/  "viewer_captureScreenshot",
        /*name*/"Capture viewport screenshot",
        /*desc*/"Render the viewer to a PNG. Default (`includeUi: true`) captures the whole window including panels, ribbon, and dialogs; set "
                "`includeUi: false` to capture only the 3D viewport. "
                "For the 3D-only mode, optional `width`/`height` request a specific resolution (zero or missing = "
                "current viewport size) and `transparentBg` (default false) omits the background — these options "
                "are ignored when `includeUi` is true (window capture always uses the current framebuffer with its "
                "normal background). "
                "Pass `filePath` to save server-side; response is `{path, width, height}`. Omit `filePath` to "
                "receive the PNG inline as an MCP image content block; `structuredContent` carries `{width, height}`. "
                "Not available in -noWindow mode.",
        /*input_schema*/Schema::Object{}
            .addMemberOpt( "includeUi",     Schema::Bool{} )
            .addMemberOpt( "width",         Schema::Number{} )
            .addMemberOpt( "height",        Schema::Number{} )
            .addMemberOpt( "transparentBg", Schema::Bool{} )
            .addMemberOpt( "filePath",      Schema::String{} ),
        /*output_schema*/Schema::Object{}
            .addMemberOpt( "path",   Schema::String{} )
            .addMember(    "width",  Schema::Number{} )
            .addMember(    "height", Schema::Number{} ),
        /*func*/mcpViewerCaptureScreenshot,
        /*readOnly*/true
    );

    server.addTool(
        /*id*/  "viewer_sendMouseEvent",
        /*name*/"Inject a mouse event",
        /*desc*/"Queue a mouse event on the viewer's event loop. `type` is one of `down`, `up`, `click`, `move`, `scroll`. "
                "`button` is `left`/`right`/`middle` (required for `down`/`up`/`click`). `x`/`y` are window pixels, "
                "top-left origin (required for `move`; optional for `down`/`up`/`click` — if given, a mouseMove is "
                "sent first so hover state resolves before the button event). `scrollDelta` is required for `scroll` "
                "(positive = scroll up). `modifiers` is an array of any of `ctrl`/`shift`/`alt`/`super`, held during this "
                "call's events, also for UI widgets (e.g. Ctrl+click in the scene list), in addition to the modifier keys "
                "held by `viewer_sendKeyboardEvent` `down` (e.g. Ctrl during a drag, until its `up`). "
                "`click` is shorthand for `down` + `up` at the same position. "
                "Waits until the UI has processed the events, at most 20 seconds; if an action they started keeps "
                "MeshInspector busy longer, returns `busy` with an explanation, see `ui_progressStatus`. "
                "Not available in -noWindow mode.",
        /*input_schema*/Schema::Object{}
            .addMember(    "type",        Schema::String{} )
            .addMemberOpt( "button",      Schema::String{} )
            .addMemberOpt( "x",           Schema::Number{} )
            .addMemberOpt( "y",           Schema::Number{} )
            .addMemberOpt( "scrollDelta", Schema::Number{} )
            .addMemberOpt( "modifiers",   Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Object{}.addMemberOpt( "busy", Schema::String{} ),
        /*func*/mcpViewerSendMouseEvent
    );

    server.addTool(
        /*id*/  "viewer_sendKeyboardEvent",
        /*name*/"Inject a keyboard event",
        /*desc*/"Queue a keyboard event on the viewer's event loop. `type` is one of `down`, `up`, `press`, `repeat`. "
                "`key` is either a single printable character (e.g. `\"a\"`, `\"A\"`, `\"5\"`) or a named key "
                "(`Escape`, `Enter`, `Space`, `Tab`, `Backspace`, `Delete`, `Home`, `End`, `PageUp`, `PageDown`, "
                "`ArrowUp`/`ArrowDown`/`ArrowLeft`/`ArrowRight`, `F1`…`F25`) or a modifier key (`Ctrl`/`Shift`/`Alt`/`Super`: "
                "the left one; `down` holds it until its `up`, for UI widgets and later mouse and key events too). "
                "`press` is for text input — it emits a "
                "unicode character event and requires a single printable character. `down`/`up`/`repeat` send raw "
                "key events (use these for shortcuts like Ctrl+S, or for keys that don't produce text). `modifiers` "
                "is an array of any of `ctrl`/`shift`/`alt`/`super`. "
                "Waits until the UI has processed the event, at most 20 seconds; if an action it started keeps "
                "MeshInspector busy longer, returns `busy` with an explanation, see `ui_progressStatus`. "
                "Not available in -noWindow mode.",
        /*input_schema*/Schema::Object{}
            .addMember(    "type",      Schema::String{} )
            .addMember(    "key",       Schema::String{} )
            .addMemberOpt( "modifiers", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Object{}.addMemberOpt( "busy", Schema::String{} ),
        /*func*/mcpViewerSendKeyboardEvent
    );

    server.addTool(
        /*id*/  "viewer_sendDragDropEvent",
        /*name*/"Inject a file drag-and-drop event",
        /*desc*/"Queue an event of files dragged from the OS over the viewer's window, as the OS sends it. `type` is one of "
                "`enter` (a drag enters the window), `over` (it moves to `x`/`y`), `leave` (it leaves without dropping), "
                "`drop` (drops `paths`, an array of file paths). `x`/`y` are window pixels, top-left origin, as in "
                "`viewer_sendMouseEvent`: required for `over`; with any other `type` an `over` to them is sent too. "
                "The drop lands where the last `over` was, e.g. on the scene list (files are added) or on the viewport "
                "(the scene is replaced); a mouse move in between forgets that position. Between `enter` and "
                "`drop`/`leave` the drop zones are highlighted. "
                "Waits until the UI has processed the events, at most 20 seconds; if an action they started keeps "
                "MeshInspector busy longer, returns `busy` with an explanation, see `ui_progressStatus`. "
                "Not available in -noWindow mode.",
        /*input_schema*/Schema::Object{}
            .addMember(    "type",  Schema::String( std::vector<std::string>{ "enter", "over", "leave", "drop" } ) )
            .addMemberOpt( "x",     Schema::Number{} )
            .addMemberOpt( "y",     Schema::Number{} )
            .addMemberOpt( "paths", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Object{}.addMemberOpt( "busy", Schema::String{} ),
        /*func*/mcpViewerSendDragDropEvent
    );

    server.addTool(
        /*id*/  "viewer_sendTouchpadEvent",
        /*name*/"Inject a touchpad gesture event",
        /*desc*/"Queue a touchpad gesture event, as a precision touchpad or a Mac trackpad sends it. `gesture` is `rotate` "
                "(two fingers turning), `swipe` (two fingers moving) or `zoom` (pinch); `phase` is `begin`, `update` or `end`: "
                "send one `begin`, any number of `update`s, one `end`. `begin` acts on the viewport under the mouse, so send a "
                "`viewer_sendMouseEvent` `move` there first. `update` takes: for `rotate`, `angle` in radians since `begin`; "
                "for `swipe`, the move `dx`/`dy` in pixels since the previous `update`; for `zoom`, `scale` relative to "
                "`begin` (1 = unchanged, above 1 zooms in). `kinetic` (default false) marks momentum `update`s sent after "
                "the fingers left the touchpad (ignored by default settings). "
                "Waits until the UI has processed the event, at most 20 seconds; if an action it started keeps "
                "MeshInspector busy longer, returns `busy` with an explanation, see `ui_progressStatus`. "
                "Not available in -noWindow mode.",
        /*input_schema*/Schema::Object{}
            .addMember(    "gesture", Schema::String( std::vector<std::string>{ "rotate", "swipe", "zoom" } ) )
            .addMember(    "phase",   Schema::String( std::vector<std::string>{ "begin", "update", "end" } ) )
            .addMemberOpt( "angle",   Schema::Number{} )
            .addMemberOpt( "dx",      Schema::Number{} )
            .addMemberOpt( "dy",      Schema::Number{} )
            .addMemberOpt( "scale",   Schema::Number{} )
            .addMemberOpt( "kinetic", Schema::Bool{} ),
        /*output_schema*/Schema::Object{}.addMemberOpt( "busy", Schema::String{} ),
        /*func*/mcpViewerSendTouchpadEvent
    );

    server.addTool(
        /*id*/  "viewer_shutdown",
        /*name*/"Close MeshInspector",
        /*desc*/"Cleanly stop MeshInspector's event loop and exit the process. Returns immediately so the MCP "
                "response can flush before the server socket closes; the actual shutdown happens on the next frame. "
                "Until the process has exited, other tools fail and the gateway's `status` still says `running`; once it "
                "says `not started`, the gateway's `launch` tool can bring MeshInspector back up. "
                "If the scene has unsaved changes, an `Application Close` modal is raised instead of exiting; "
                "check `ui_listEntries` for the modal and dismiss it with the `Cancel` / `Don't Save` / `Save` buttons. "
                "Pass `force: true` to discard unsaved changes and exit without the modal.",
        /*input_schema*/Schema::Object{}.addMemberOpt( "force", Schema::Bool{} ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpViewerShutdown
    );
}; // MR_ON_INIT

} // namespace MR::Mcp

#endif
