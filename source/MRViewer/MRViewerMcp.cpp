#ifndef MESHLIB_NO_MCP

#include "MRMcp/MRMcp.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBase64.h"
#include "MRMesh/MRBox.h"
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
#include "MRViewer/MRViewer.h"
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
    throw std::runtime_error( fmt::format( "Unknown key `{}` (use a single printable char or a name like `Escape`, `Enter`, `ArrowUp`, `F5`, `Num7`, `Delete`, `ForwardDelete`)", s ) );
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
    const std::string type = args.at( "type" ).get<std::string>();
    const int mods = parseModifiers( args );
    auto& v = getViewerInstance();

    // mouseMove between hover-resolve and click matters for many plugins; follow the pattern used
    // by Python UI tests in test_ui/scenarios/scenario_helpers.py.
    const bool hasXY = args.contains( "x" ) && args.contains( "y" );
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
            v.emplaceEvent( "mcp mouseDown", [&v, btn, mods]{ v.mouseDown( btn, mods ); } );
            waitFrames();
        }
        if ( type == "up" || type == "click" )
        {
            v.emplaceEvent( "mcp mouseUp", [&v, btn, mods]{ v.mouseUp( btn, mods ); } );
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
                "receive the PNG inline as an MCP image content block; `structuredContent` carries `{width, height}`.",
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
                "(positive = scroll up). `modifiers` is an array of any of `ctrl`/`shift`/`alt`/`super`. "
                "`click` is shorthand for `down` + `up` at the same position. "
                "Waits until the UI has processed the events, at most 20 seconds; if an action they started keeps "
                "MeshInspector busy longer, returns `busy` with an explanation, see `ui_progressStatus`.",
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
                "`ArrowUp`/`ArrowDown`/`ArrowLeft`/`ArrowRight`, `F1`…`F25`). `press` is for text input — it emits a "
                "unicode character event and requires a single printable character. `down`/`up`/`repeat` send raw "
                "key events (use these for shortcuts like Ctrl+S, or for keys that don't produce text). `modifiers` "
                "is an array of any of `ctrl`/`shift`/`alt`/`super`. "
                "Waits until the UI has processed the event, at most 20 seconds; if an action it started keeps "
                "MeshInspector busy longer, returns `busy` with an explanation, see `ui_progressStatus`.",
        /*input_schema*/Schema::Object{}
            .addMember(    "type",      Schema::String{} )
            .addMember(    "key",       Schema::String{} )
            .addMemberOpt( "modifiers", Schema::Array( Schema::String{} ) ),
        /*output_schema*/Schema::Object{}.addMemberOpt( "busy", Schema::String{} ),
        /*func*/mcpViewerSendKeyboardEvent
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
