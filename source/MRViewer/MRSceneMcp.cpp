#ifndef MESHLIB_NO_MCP

#include "MRMcp/MRMcp.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBase64.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRChangeColoringActions.h"
#include "MRMesh/MRChangeNameAction.h"
#include "MRMesh/MRChangeObjectFields.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRMatrix3Decompose.h"
#include "MRMesh/MROnInit.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectLinesHolder.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRObjectPointsHolder.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRMesh/MRViewportProperty.h"
#include "MRMesh/MRVisualObject.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRMcpCommon.h"

#include <cstdint>
#include <fstream>
#include <memory>

namespace MR::Mcp
{

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
    return {
        .translation = xf.b,
        .rotationDeg = rotation.toEulerAngles() * ( 180.f / PI_F ),
        .scale       = { scaling.x.x, scaling.y.y, scaling.z.z },
    };
}

// Inverse of decomposeXf: build an AffineXf3f from translation + XYZ-Euler degrees + scale.
static AffineXf3f composeXf( const DecomposedXf& d )
{
    AffineXf3f xf;
    xf.A = Matrix3f::rotationFromEuler( d.rotationDeg * ( PI_F / 180.f ) ) * Matrix3f::scale( d.scale );
    xf.b = d.translation;
    return xf;
}

// Parse the `transform` subobject accepted by scene.setObjectState. Every field is optional;
// omitted components fall back to identity (translation=0, rotation=0, scale=1). `scale` may be
// a scalar (applies uniformly) or a 3-element array.
static DecomposedXf parseTransform( const nlohmann::json& t )
{
    DecomposedXf d{ .translation = {}, .rotationDeg = {}, .scale = { 1.f, 1.f, 1.f } };
    const auto readVec3 = []( const nlohmann::json& j )
    {
        return Vector3f{ j[0].get<float>(), j[1].get<float>(), j[2].get<float>() };
    };
    if ( t.contains( "translation" ) )
        d.translation = readVec3( t["translation"] );
    if ( t.contains( "rotation" ) )
        d.rotationDeg = readVec3( t["rotation"] );
    if ( t.contains( "scale" ) )
    {
        const auto& s = t["scale"];
        d.scale = s.is_number() ? Vector3f::diagonal( s.get<float>() ) : readVec3( s );
    }
    return d;
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

// Color in/out as `[r, g, b]` or `[r, g, b, a]` floats in [0, 1].
static nlohmann::json colorToJson( const Color& c )
{
    return nlohmann::json::array( { c.r / 255.f, c.g / 255.f, c.b / 255.f, c.a / 255.f } );
}

static Color parseColor( const nlohmann::json& j, std::string_view field )
{
    if ( !j.is_array() || ( j.size() != 3 && j.size() != 4 ) )
        throw std::runtime_error( fmt::format( "`{}` must be a 3- or 4-element array of floats in [0,1]", field ) );
    const float r = j[0].get<float>();
    const float g = j[1].get<float>();
    const float b = j[2].get<float>();
    const float a = j.size() == 4 ? j[3].get<float>() : 1.0f;
    return Color{ r, g, b, a }; // the float ctor clamps to [0, 1]
}

// Per-object visualization properties (colors, alpha, size, flags). Fields only present if the
// object type supports them.
static nlohmann::json visualizationToJson( const Object& obj )
{
    nlohmann::json viz = nlohmann::json::object();
    const auto* vo = dynamic_cast<const VisualObject*>( &obj );
    if ( !vo )
        return viz;

    viz["selectedColor"]   = colorToJson( vo->getFrontColor( true ) );
    viz["unselectedColor"] = colorToJson( vo->getFrontColor( false ) );
    viz["globalAlpha"]     = vo->getGlobalAlpha() / 255.f;

    if ( const auto* mh = dynamic_cast<const ObjectMeshHolder*>( &obj ) )
    {
        viz["flatShading"] = mh->getVisualizeProperty( MeshVisualizePropertyType::FlatShading, ViewportMask::any() );
        viz["showEdges"]   = mh->getVisualizeProperty( MeshVisualizePropertyType::Edges, ViewportMask::any() );
        viz["edgeWidth"]   = mh->getEdgeWidth();
    }
    else if ( const auto* lh = dynamic_cast<const ObjectLinesHolder*>( &obj ) )
    {
        viz["edgeWidth"] = lh->getLineWidth();
    }
    if ( const auto* ph = dynamic_cast<const ObjectPointsHolder*>( &obj ) )
    {
        viz["pointSize"] = ph->getPointSize();
    }
    return viz;
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

        // Start with the object's own info lines (same as MI's Information panel), then mirror
        // the structured `transform` field in text form too — LLMs often skim `info` as the
        // human-readable summary. Derived from `transformToJson`'s output so the text can't
        // drift from JSON.
        std::vector<std::string> info = obj->getInfoLines();
        const auto tfJson = transformToJson( obj->xf() );
        const auto vec3Line = []( std::string_view label, const nlohmann::json& v )
        {
            return fmt::format( "{}: ({}, {}, {})", label,
                v[0].get<float>(), v[1].get<float>(), v[2].get<float>() );
        };
        info.push_back( vec3Line( "translation",  tfJson["translation"] ) );
        info.push_back( vec3Line( "rotation deg", tfJson["rotation"] ) );
        info.push_back( vec3Line( "scale",        tfJson["scale"] ) );
        out["info"] = std::move( info );

        out["visualization"] = visualizationToJson( *obj );
    } );
    return out;
}

static nlohmann::json mcpSceneSetObjectState( const nlohmann::json& args )
{
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto objEx = resolveId( args.at( "id" ).get<uint64_t>() );
        if ( !objEx )
            throw std::runtime_error( objEx.error() );
        auto obj = *objEx;

        SCOPED_HISTORY( _t( "MCP setObjectState" ) );
        if ( args.contains( "name" ) )
        {
            AppendHistory<ChangeNameAction>( _t( "rename" ), obj );
            obj->setName( args["name"].get<std::string>() );
        }
        if ( args.contains( "visible" ) )
        {
            AppendHistory<ChangeObjectVisibilityAction>( _t( "visibility" ), obj );
            obj->setVisible( args["visible"].get<bool>(), ViewportMask::all() );
        }
        if ( args.contains( "selected" ) )
        {
            AppendHistory<ChangeObjectSelectedAction>( _t( "selection" ), obj );
            obj->select( args["selected"].get<bool>() );
        }
        if ( args.contains( "transform" ) )
        {
            AppendHistory<ChangeXfAction>( _t( "transform" ), obj );
            obj->setXf( composeXf( parseTransform( args["transform"] ) ) );
        }
        if ( args.contains( "visualization" ) )
        {
            const auto& viz = args["visualization"];

            auto visObj = std::dynamic_pointer_cast<VisualObject>( obj );
            const bool touchesVisProps =
                viz.contains( "selectedColor" ) || viz.contains( "unselectedColor" ) || viz.contains( "globalAlpha" );
            if ( touchesVisProps && !visObj )
                throw std::runtime_error( "Object does not support color/alpha visualization properties." );

            if ( viz.contains( "selectedColor" ) )
            {
                const Color c = parseColor( viz["selectedColor"], "selectedColor" );
                AppendHistory<ChangeObjectColorAction>( _t( "selected color" ), visObj, ChangeObjectColorAction::Type::Selected );
                visObj->setFrontColorsForAllViewports( ViewportProperty<Color>{ c }, true );
            }
            if ( viz.contains( "unselectedColor" ) )
            {
                const Color c = parseColor( viz["unselectedColor"], "unselectedColor" );
                AppendHistory<ChangeObjectColorAction>( _t( "unselected color" ), visObj, ChangeObjectColorAction::Type::Unselected );
                visObj->setFrontColorsForAllViewports( ViewportProperty<Color>{ c }, false );
            }
            if ( viz.contains( "globalAlpha" ) )
            {
                const float a = viz["globalAlpha"].get<float>();
                if ( !std::isfinite( a ) || a < 0.f || a > 1.f )
                    throw std::runtime_error( "`globalAlpha` must be in [0, 1]" );
                // No dedicated history action for alpha yet — note: Ctrl+Z won't revert this alone.
                visObj->setGlobalAlphaForAllViewports( ViewportProperty<uint8_t>{ uint8_t( a * 255.f + 0.5f ) } );
            }

            auto meshHolder   = std::dynamic_pointer_cast<ObjectMeshHolder>( obj );
            auto linesHolder  = std::dynamic_pointer_cast<ObjectLinesHolder>( obj );
            auto pointsHolder = std::dynamic_pointer_cast<ObjectPointsHolder>( obj );

            if ( viz.contains( "flatShading" ) )
            {
                if ( !meshHolder )
                    throw std::runtime_error( "`flatShading` is only available on mesh objects." );
                const bool v = viz["flatShading"].get<bool>();
                AppendHistory<ChangeVisualizePropertyAction>( _t( "flat shading" ), meshHolder, AnyVisualizeMaskEnum{ MeshVisualizePropertyType::FlatShading } );
                meshHolder->setVisualizeProperty( v, MeshVisualizePropertyType::FlatShading, ViewportMask::all() );
            }
            if ( viz.contains( "showEdges" ) )
            {
                if ( !meshHolder )
                    throw std::runtime_error( "`showEdges` is only available on mesh objects." );
                const bool v = viz["showEdges"].get<bool>();
                AppendHistory<ChangeVisualizePropertyAction>( _t( "show edges" ), meshHolder, AnyVisualizeMaskEnum{ MeshVisualizePropertyType::Edges } );
                meshHolder->setVisualizeProperty( v, MeshVisualizePropertyType::Edges, ViewportMask::all() );
            }
            if ( viz.contains( "edgeWidth" ) )
            {
                const float w = viz["edgeWidth"].get<float>();
                if ( !std::isfinite( w ) || w < 0.f )
                    throw std::runtime_error( "`edgeWidth` must be a non-negative finite number." );
                // No dedicated history action yet — Ctrl+Z won't revert this alone.
                if ( meshHolder )
                    meshHolder->setEdgeWidth( w );
                else if ( linesHolder )
                    linesHolder->setLineWidth( w );
                else
                    throw std::runtime_error( "`edgeWidth` is only available on mesh or polyline objects." );
            }
            if ( viz.contains( "pointSize" ) )
            {
                if ( !pointsHolder )
                    throw std::runtime_error( "`pointSize` is only available on point-cloud objects." );
                const float s = viz["pointSize"].get<float>();
                if ( !std::isfinite( s ) || s < 0.f )
                    throw std::runtime_error( "`pointSize` must be a non-negative finite number." );
                // No dedicated history action yet — Ctrl+Z won't revert this alone.
                pointsHolder->setPointSize( s );
            }
        }
    } );
    skipFramesAfterInput();
    return nlohmann::json::object();
}

static nlohmann::json mcpSceneRemoveObject( const nlohmann::json& args )
{
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto objEx = resolveId( args.at( "id" ).get<uint64_t>() );
        if ( !objEx )
            throw std::runtime_error( objEx.error() );
        auto obj = *objEx;

        AppendHistory<ChangeSceneAction>( _t( "MCP removeObject" ), obj, ChangeSceneAction::Type::RemoveObject );
        obj->detachFromParent();
    } );
    skipFramesAfterInput();
    return nlohmann::json::object();
}

static nlohmann::json mcpSceneGetObject( const nlohmann::json& args )
{
    nlohmann::json out = nlohmann::json::object();
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto objEx = resolveId( args.at( "id" ).get<uint64_t>() );
        if ( !objEx )
            throw std::runtime_error( objEx.error() );
        auto obj = *objEx;

        if ( args.contains( "filePath" ) )
        {
            const auto path = pathFromUtf8( args["filePath"].get<std::string>() );
            auto saved = ObjectSave::toAnySupportedFormat( *obj, path );
            if ( !saved )
                throw std::runtime_error( saved.error() );
            out["path"] = utf8string( path );
            return;
        }
        if ( args.contains( "extension" ) )
        {
            std::string ext = args["extension"].get<std::string>();
            if ( !ext.empty() && ext.front() != '.' )
                ext.insert( ext.begin(), '.' );

            UniqueTemporaryFolder tmp{};
            const std::filesystem::path path = tmp / pathFromUtf8( obj->name() + ext );
            auto saved = ObjectSave::toAnySupportedFormat( *obj, path );
            if ( !saved )
                throw std::runtime_error( saved.error() );
            std::ifstream in( path, std::ios::binary );
            if ( !in )
                throw std::runtime_error( fmt::format( "Could not read back temp file {}", utf8string( path ) ) );
            std::vector<std::uint8_t> bytes( ( std::istreambuf_iterator<char>( in ) ), std::istreambuf_iterator<char>() );
            out["bytes"] = encode64( bytes.data(), bytes.size() );
            return;
        }
        throw std::runtime_error( "scene.getObject requires either `filePath` or `extension`." );
    } );
    return out;
}

static nlohmann::json mcpSceneAddObject( const nlohmann::json& args )
{
    std::vector<uint64_t> addedIds;
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        std::shared_ptr<Object> parentHolder;
        Object* parent = &SceneRoot::get();
        if ( args.contains( "parentId" ) && args["parentId"].get<uint64_t>() != 0 )
        {
            auto pEx = resolveId( args["parentId"].get<uint64_t>() );
            if ( !pEx )
                throw std::runtime_error( pEx.error() );
            parentHolder = *pEx;
            parent = parentHolder.get();
        }

        std::filesystem::path path;
        std::optional<UniqueTemporaryFolder> tmp;
        std::optional<std::string> overrideName;

        if ( args.contains( "filePath" ) )
        {
            path = pathFromUtf8( args["filePath"].get<std::string>() );
        }
        else if ( args.contains( "bytes" ) )
        {
            std::string ext = args.at( "extension" ).get<std::string>();
            if ( !ext.empty() && ext.front() != '.' )
                ext.insert( ext.begin(), '.' );
            overrideName = args.at( "name" ).get<std::string>();
            const auto decoded = decode64( args["bytes"].get<std::string>() );

            tmp.emplace();
            path = *tmp / pathFromUtf8( *overrideName + ext );
            std::ofstream out( path, std::ios::binary );
            if ( !out )
                throw std::runtime_error( fmt::format( "Could not write temp file {}", utf8string( path ) ) );
            out.write( reinterpret_cast<const char*>( decoded.data() ), std::streamsize( decoded.size() ) );
        }
        else
        {
            throw std::runtime_error( "scene.addObject requires either `filePath` or `bytes`+`extension`+`name`." );
        }

        auto loaded = loadObjectFromFile( path );
        if ( !loaded )
            throw std::runtime_error( loaded.error() );

        SCOPED_HISTORY( _t( "MCP addObject" ) );
        for ( auto& obj : loaded->objs )
        {
            if ( overrideName )
                obj->setName( *overrideName );
            AppendHistory<ChangeSceneAction>( _t( "add" ), obj, ChangeSceneAction::Type::AddObject );
            parent->addChild( obj );
            addedIds.push_back( idOf( obj.get() ) );
        }
    } );
    skipFramesAfterInput();
    return nlohmann::json::object( { { "ids", addedIds } } );
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
                "children), `worldBox` (axis-aligned world-space bounding box), `info` (human-readable property "
                "lines - e.g. vertex/face counts, area - from the UI's info panel), and `visualization` (current "
                "render-time properties — colors, alpha, size, flags; see `scene.setObjectState` for field semantics; "
                "fields that don't apply to this object's type are omitted).",
        /*input_schema*/Schema::Object{}.addMember( "id", Schema::Number{} ),
        /*output_schema*/entrySchema()
            .addMember( "childIds", Schema::Array( Schema::Number{} ) )
            .addMember( "worldBox", Schema::Object{}
                .addMember( "min", Schema::Array( Schema::Number{} ) )
                .addMember( "max", Schema::Array( Schema::Number{} ) ) )
            .addMember( "info",     Schema::Array( Schema::String{} ) )
            .addMember( "visualization", Schema::Object{} ),
        /*func*/mcpSceneGetObjectInfo
    );

    server.addTool(
        /*id*/  "scene.setObjectState",
        /*name*/"Set scene object state",
        /*desc*/std::string( kSceneIdSemantics ) +
                "Mutate an existing object. Every field other than `id` is optional; absent fields leave state alone. "
                "`transform` replaces the full xf and is itself composed from optional `translation` / `rotation` / `scale` "
                "(missing components default to identity; `rotation` is XYZ-Euler degrees; `scale` accepts a scalar or a "
                "3-element array). `visualization` updates render-time properties: `selectedColor` / `unselectedColor` as "
                "`[r,g,b]` or `[r,g,b,a]` floats in [0,1]; `globalAlpha` as a float in [0,1]; `flatShading` / `showEdges` "
                "as bools (mesh objects only); `edgeWidth` as a non-negative float (meshes use it for edge width, "
                "polylines for line width); `pointSize` as a non-negative float (point-cloud objects only). Fields "
                "not applicable to the object's type return a typed error. "
                "A single call that touches multiple fields is wrapped in one history entry, so Ctrl+Z reverts it "
                "atomically (note: `globalAlpha`, `edgeWidth`, and `pointSize` do not have dedicated history actions "
                "yet — they are applied but the undo step will skip them).",
        /*input_schema*/Schema::Object{}
            .addMember(    "id",        Schema::Number{} )
            .addMemberOpt( "name",      Schema::String{} )
            .addMemberOpt( "visible",   Schema::Bool{} )
            .addMemberOpt( "selected",  Schema::Bool{} )
            .addMemberOpt( "transform", transformSchema() )
            .addMemberOpt( "visualization", Schema::Object{}
                .addMemberOpt( "selectedColor",   Schema::Array( Schema::Number{} ) )
                .addMemberOpt( "unselectedColor", Schema::Array( Schema::Number{} ) )
                .addMemberOpt( "globalAlpha",     Schema::Number{} )
                .addMemberOpt( "flatShading",     Schema::Bool{} )
                .addMemberOpt( "showEdges",       Schema::Bool{} )
                .addMemberOpt( "edgeWidth",       Schema::Number{} )
                .addMemberOpt( "pointSize",       Schema::Number{} ) ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpSceneSetObjectState
    );

    server.addTool(
        /*id*/  "scene.removeObject",
        /*name*/"Remove scene object",
        /*desc*/std::string( kSceneIdSemantics ) +
                "Detach the object from its parent (recursive — removing a group removes its subtree). Undoable.",
        /*input_schema*/Schema::Object{}.addMember( "id", Schema::Number{} ),
        /*output_schema*/Schema::Object{},
        /*func*/mcpSceneRemoveObject
    );

    server.addTool(
        /*id*/  "scene.getObject",
        /*name*/"Serialize scene object (to file or inline bytes)",
        /*desc*/std::string( kSceneIdSemantics ) +
                "Serialize an object to an STL/OBJ/PLY/etc. format. Pass either `filePath` (server-side path; "
                "format selected by extension) — response is `{path: <written-path>}`; or `extension` (e.g. `\"stl\"`) — "
                "response is `{bytes: <base64 payload>}`.",
        /*input_schema*/Schema::Object{}
            .addMember(    "id",        Schema::Number{} )
            .addMemberOpt( "filePath",  Schema::String{} )
            .addMemberOpt( "extension", Schema::String{} ),
        /*output_schema*/Schema::Object{}
            .addMemberOpt( "path",  Schema::String{} )
            .addMemberOpt( "bytes", Schema::String{} ),
        /*func*/mcpSceneGetObject
    );

    server.addTool(
        /*id*/  "scene.addObject",
        /*name*/"Add scene object from file or bytes",
        /*desc*/std::string( kSceneIdSemantics ) +
                "Load one or more objects into the scene. Pass either `filePath` (server-side path to an STL/OBJ/PLY/etc. "
                "file), or `bytes` (base64 payload) + `extension` (e.g. `\"stl\"`) + `name` (display name). Optional "
                "`parentId` (default = scene root). Returns `{ids: [uint64...]}` — loaders can produce multiple objects "
                "for scene files (.mru, glTF). Undoable.",
        /*input_schema*/Schema::Object{}
            .addMemberOpt( "filePath",  Schema::String{} )
            .addMemberOpt( "bytes",     Schema::String{} )
            .addMemberOpt( "extension", Schema::String{} )
            .addMemberOpt( "name",      Schema::String{} )
            .addMemberOpt( "parentId",  Schema::Number{} ),
        /*output_schema*/Schema::Object{}.addMember( "ids", Schema::Array( Schema::Number{} ) ),
        /*func*/mcpSceneAddObject
    );
}; // MR_ON_INIT

} // namespace MR::Mcp

#endif
