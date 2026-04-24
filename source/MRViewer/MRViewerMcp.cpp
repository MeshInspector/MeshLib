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
#include "MRViewer/MRMcpCommon.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"

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

} // namespace

static nlohmann::json mcpViewerFit( const nlohmann::json& args )
{
    const float factor = args.value( "factor", 1.0f );
    const bool snapView = args.value( "snapView", true );

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
            vp.fitData( factor, snapView );
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
        for ( const Vector3f& p : points )
            box.include( p );
        if ( !box.valid() )
            throw std::runtime_error( "viewer.fit: the given objects and points contribute no bounding volume." );
        vp.fitBox( box, factor, snapView );
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
        vp.fitData();
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
    }
    out["width"] = img.resolution.x;
    out["height"] = img.resolution.y;
    return nlohmann::json::object( { { "result", std::move( out ) } } );
}

MR_ON_INIT{
    Server& server = getDefaultServer();

    server.addTool(
        /*id*/  "viewer.fit",
        /*name*/"Fit camera to scene or a subset",
        /*desc*/"Frame a set of objects and/or world-space points in the active viewport. Pass `objectIds` (scene-object "
                "ids from `scene.listObjectTree`) and/or `points` (3-element `[x,y,z]` arrays). If both are present, "
                "their bounding volumes are unioned. If neither is given, fits the whole scene (same as the UI's "
                "\"Fit\" action). `factor` (default 1.0) controls framing margin — higher means more screen coverage; "
                "`snapView` (default true) snaps the camera to the nearest canonical quaternion.",
        /*input_schema*/Schema::Object{}
            .addMemberOpt( "objectIds", Schema::Array( Schema::Number{} ) )
            .addMemberOpt( "points",    Schema::Array( Schema::Array( Schema::Number{} ) ) )
            .addMemberOpt( "factor",    Schema::Number{} )
            .addMemberOpt( "snapView",  Schema::Bool{} ),
        /*output_schema*/Schema::Empty{},
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
        /*output_schema*/Schema::Empty{},
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
}; // MR_ON_INIT

} // namespace MR::Mcp

#endif
