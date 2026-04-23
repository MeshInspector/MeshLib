#ifndef MESHLIB_NO_MCP

#include "MRMcp/MRMcp.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRMatrix3Decompose.h"
#include "MRMesh/MROnInit.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRMcpCommon.h"

#include <cstdint>
#include <memory>

namespace MR::Mcp
{

// Pointer -> id (as a JSON number). Matches the `##<id>` suffix in ImGui labels for the same object.
static uint64_t idOf( const Object* obj )
{
    return static_cast<uint64_t>( reinterpret_cast<uintptr_t>( obj ) );
}

// Resolve a raw-pointer id (as returned by scene.listObjectTree) to a shared_ptr<Object>, verifying
// the object is still reachable from SceneRoot. Defends against stale ids into freed memory.
static Expected<std::shared_ptr<Object>> resolveId( uint64_t id )
{
    auto* asPtr = reinterpret_cast<Object*>( static_cast<uintptr_t>( id ) );
    if ( !asPtr )
        return unexpected( "Object id 0 is not valid here." );

    for ( const auto& obj : getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selectable ) )
    {
        if ( obj.get() == asPtr )
            return obj;
    }
    return unexpected( fmt::format( "No object with id {}. Call scene.listObjectTree to enumerate.", id ) );
}

// Pick the root for a listObjectTree / getObjectInfo call: the supplied id, or SceneRoot if 0/absent.
static Expected<Object*> pickRoot( const nlohmann::json& args )
{
    const uint64_t id = args.value( "rootId", uint64_t( 0 ) );
    if ( id == 0 )
        return &SceneRoot::get();
    auto resolved = resolveId( id );
    if ( !resolved )
        return unexpected( resolved.error() );
    return resolved->get();
}

// Same translation/rotation/scale shape that `scene.setObjectState` accepts. Rotation is XYZ-Euler
// in degrees. Agents can read-modify-write without matrix math.
struct DecomposedXf { Vector3f translation, rotationDeg, scale; };

static DecomposedXf decomposeXf( const AffineXf3f& xf )
{
    Matrix3f rotation, scaling;
    decomposeMatrix3( xf.A, rotation, scaling );
    constexpr float kRadToDeg = 57.29577951308232f; // 180 / pi
    return {
        .translation = xf.b,
        .rotationDeg = rotation.toEulerAngles() * kRadToDeg,
        .scale       = { scaling.x.x, scaling.y.y, scaling.z.z },
    };
}

static nlohmann::json transformToJson( const AffineXf3f& xf )
{
    const auto d = decomposeXf( xf );
    return nlohmann::json::object( {
        { "translation", nlohmann::json::array( { d.translation.x, d.translation.y, d.translation.z } ) },
        { "rotation",    nlohmann::json::array( { d.rotationDeg.x, d.rotationDeg.y, d.rotationDeg.z } ) },
        { "scale",       nlohmann::json::array( { d.scale.x,       d.scale.y,       d.scale.z } ) },
    } );
}

static nlohmann::json objectRow( const Object& obj, const Object* parent )
{
    return nlohmann::json::object( {
        { "id",        idOf( &obj ) },
        { "parentId",  idOf( parent ) },
        { "name",      obj.name() },
        { "type",      obj.className() },
        { "selected",  obj.isSelected() },
        { "visible",   obj.isVisible( ViewportMask::any() ) },
        { "transform", transformToJson( obj.xf() ) },
    } );
}

static nlohmann::json mcpSceneListObjectTree( const nlohmann::json& args )
{
    nlohmann::json rows = nlohmann::json::array();
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto rootEx = pickRoot( args );
        if ( !rootEx )
            throw std::runtime_error( rootEx.error() );
        Object* root = *rootEx;

        for ( const auto& obj : getAllObjectsInTree<Object>( root, ObjectSelectivityType::Selectable ) )
            rows.push_back( objectRow( *obj, obj->parent() ) );
    } );
    return nlohmann::json::object( { { "result", std::move( rows ) } } );
}

static nlohmann::json mcpSceneGetObjectInfo( const nlohmann::json& args )
{
    nlohmann::json out;
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        const uint64_t id = args.at( "id" ).get<uint64_t>();
        auto objEx = resolveId( id );
        if ( !objEx )
            throw std::runtime_error( objEx.error() );
        const auto& obj = *objEx;

        out = objectRow( *obj, obj->parent() );

        nlohmann::json childIds = nlohmann::json::array();
        for ( const auto& child : obj->children() )
            childIds.push_back( idOf( child.get() ) );
        out["childIds"] = std::move( childIds );

        const Box3f box = obj->getWorldBox();
        out["worldBox"] = nlohmann::json::object( {
            { "min", nlohmann::json::array( { box.min.x, box.min.y, box.min.z } ) },
            { "max", nlohmann::json::array( { box.max.x, box.max.y, box.max.z } ) },
        } );

        // Start with the object's own info lines (same as MI's Information panel),
        // then append the decomposed transform so it's also readable in plain text.
        std::vector<std::string> info = obj->getInfoLines();
        const auto d = decomposeXf( obj->xf() );
        info.push_back( fmt::format( "translation: ({}, {}, {})", d.translation.x, d.translation.y, d.translation.z ) );
        info.push_back( fmt::format( "rotation deg: ({}, {}, {})", d.rotationDeg.x, d.rotationDeg.y, d.rotationDeg.z ) );
        info.push_back( fmt::format( "scale: ({}, {}, {})",       d.scale.x,       d.scale.y,       d.scale.z ) );
        out["info"] = std::move( info );
    } );
    return nlohmann::json::object( { { "result", std::move( out ) } } );
}

static constexpr std::string_view kSceneIdSemantics =
    "`id`: opaque uint64 from `scene.listObjectTree` / `scene.getObjectInfo`. Same integer appears as the `##<id>` "
    "suffix in ImGui labels for the object, so it's usable across `scene.*` and `ui.*`. "
    "Invalidated when the object is removed; subsequent calls with a stale id return a typed error.\n\n";

MR_ON_INIT{
    Server& server = getDefaultServer();

    static const auto transformSchema = []
    {
        return Schema::Object{}
            .addMember( "translation", Schema::Array( Schema::Number{} ) )
            .addMember( "rotation",    Schema::Array( Schema::Number{} ) )
            .addMember( "scale",       Schema::Array( Schema::Number{} ) );
    };
    static const auto entrySchema = []
    {
        return Schema::Object{}
            .addMember( "id",        Schema::Number{} )
            .addMember( "parentId",  Schema::Number{} )
            .addMember( "name",      Schema::String{} )
            .addMember( "type",      Schema::String{} )
            .addMember( "selected",  Schema::Bool{} )
            .addMember( "visible",   Schema::Bool{} )
            .addMember( "transform", transformSchema() );
    };

    server.addTool(
        /*id*/  "scene.listObjectTree",
        /*name*/"List scene objects (subtree)",
        /*desc*/std::string( kSceneIdSemantics ) +
                "Flat depth-first list of every descendant of `rootId` (root itself excluded; pass `rootId = 0` or "
                "omit to walk from the scene root). Each row carries its own `parentId`, so the tree structure is "
                "recoverable. `transform` decomposes the object's xf into `{translation:[x,y,z], rotation:[rx,ry,rz], "
                "scale:[sx,sy,sz]}` (rotation is XYZ-Euler in degrees) — same shape `scene.setObjectState` accepts. "
                "`type` values include `\"Mesh\"`, `\"Point Cloud\"`, `\"Polyline\"`, `\"Voxel Volume\"`.",
        /*input_schema*/Schema::Object{}.addMemberOpt( "rootId", Schema::Number{} ),
        /*output_schema*/Schema::Array( entrySchema() ),
        /*func*/mcpSceneListObjectTree
    );

    server.addTool(
        /*id*/  "scene.getObjectInfo",
        /*name*/"Get scene object info",
        /*desc*/std::string( kSceneIdSemantics ) +
                "Return a single object's row (same shape as `scene.listObjectTree` entries) plus `childIds` (direct "
                "children), `worldBox` (axis-aligned world-space bounding box), and `info` (human-readable property "
                "lines - e.g. vertex/face counts, area - from the UI's info panel).",
        /*input_schema*/Schema::Object{}.addMember( "id", Schema::Number{} ),
        /*output_schema*/entrySchema()
            .addMember( "childIds", Schema::Array( Schema::Number{} ) )
            .addMember( "worldBox", Schema::Object{}
                .addMember( "min", Schema::Array( Schema::Number{} ) )
                .addMember( "max", Schema::Array( Schema::Number{} ) ) )
            .addMember( "info",     Schema::Array( Schema::String{} ) ),
        /*func*/mcpSceneGetObjectInfo
    );
}; // MR_ON_INIT

} // namespace MR::Mcp

#endif
