#include "MRMesh/MREmbeddedPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBoolean.h"
#include "MRMesh/MRVDBConversions.h"
#include "MRMesh/MRMeshFillHole.h"
#include "MRMesh/MRMeshMetrics.h"
#include "MRMesh/MRTunnelDetector.h"
#include "MRMesh/MRSymbolMesh.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRFixUndercuts.h"
#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRMeshDistance.h"
#include "MRMesh/MRMeshRelax.h"
#include "MRMesh/MRSurfacePath.h"
#include "MRMesh/MRPlanarPath.h"
#include "MRMesh/MRMeshRelax.h"
#include "MRMesh/MRMeshDelone.h"
#include "MRMesh/MRMeshSubdivide.h"
#include "MRMesh/MRMeshComponents.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRPositionVertsSmoothly.h"
#include "MRMesh/MRAlignTextToMesh.h"
#include "MRMesh/MRFaceFace.h"
#include "MRMesh/MRLaplacian.h"
#include "MRMesh/MRMeshDelete.h"
#include <tl/expected.hpp>

using namespace MR;

// Fix self-intersections
void fixSelfIntersections( Mesh& mesh1, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    mesh1 = convert(gridA);
}
MR_ADD_PYTHON_FUNCTION( mrmeshpy, fix_self_intersections, &fixSelfIntersections, "subtract second mesh from the first one" )

// Boolean
void booleanSub( Mesh& mesh1, const Mesh& mesh2, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    auto gridB = convert(mesh2);
    gridA -= gridB;
    mesh1 = convert(gridA);
}
MR_ADD_PYTHON_FUNCTION( mrmeshpy, boolean_sub, &booleanSub, "subtract second mesh from the first one" )

void booleanUnion( Mesh& mesh1, const Mesh& mesh2, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    auto gridB = convert(mesh2);
    gridA += gridB;
    mesh1 = convert(gridA);
}
MR_ADD_PYTHON_FUNCTION( mrmeshpy, boolean_union, &booleanUnion, "merge second mesh into the first one" )

void booleanIntersect( Mesh& mesh1, const Mesh& mesh2, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    auto gridB = convert(mesh2);
    gridA *= gridB;
    mesh1 = convert(gridA);
}
MR_ADD_PYTHON_FUNCTION( mrmeshpy, boolean_intersect, &booleanIntersect, "stores intersection of two meshes into the first one" )

// Stitch two Holes
void pythonSetStitchHolesEdgeLengthMetric( MR::StitchHolesParams& params, const Mesh& mesh )
{
    params.metric = std::make_unique<EdgeLengthStitchMetric>( mesh );
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, StitchHolesParams, [] ( pybind11::module_& m )
{
    pybind11::class_<StitchHolesParams>( m, "StitchHolesParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "outNewFaces", &StitchHolesParams::outNewFaces);

    m.def( "set_stitch_holes_metric_edge_length", pythonSetStitchHolesEdgeLengthMetric, "set edge length metric to stitch holes parameters" );
} )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, stitch_holes,
    static_cast<void ( * )(Mesh&, EdgeId, EdgeId, const StitchHolesParams&)> (&buildCylinderBetweenTwoHoles),
    "stitches two holes with presented edges on the mesh" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, stitch_two_holes,
    static_cast<bool ( * )(Mesh&, const StitchHolesParams&)> (&buildCylinderBetweenTwoHoles),
    "stitches holes on the mesh with exact two holes" )

// Fix Tunnels
FaceBitSet detectTunnelFacesMesh (const Mesh& mesh, float maxLength)
{
    return detectTunnelFaces(mesh, maxLength);
}
MR_ADD_PYTHON_FUNCTION( mrmeshpy, get_tunnel_faces, &detectTunnelFacesMesh,
    "returns tunnel faces. Remove them and stitch new holes to fill tunnels. Tunnel length is the treshold for big holes" )

// Homology Basis
MR_ADD_PYTHON_FUNCTION( mrmeshpy, detect_basis_tunnels, &MR::detectBasisTunnels, "finds homology basis as the set of edge loops" )

// Text Mesh
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SymbolMeshParams, [] ( pybind11::module_& m )
{
    pybind11::class_<SymbolMeshParams>( m, "SymbolMeshParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "text", &SymbolMeshParams::text ).
        def_readwrite( "fontDetalization", &SymbolMeshParams::fontDetalization ).
        def_readwrite( "symbolsDistanceAdditionalOffset", &SymbolMeshParams::symbolsDistanceAdditionalOffset ).
        def_readwrite( "pathToFontFile", &TextMeshAlignParams::pathToFontFile );
} )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, create_text_mesh, &createSymbolsMesh, "create text mesh. Use empty path for default" )

// Text on Mesh
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, TextAlignParams, [] ( pybind11::module_& m )
{
    pybind11::class_<EdgeIdAndCoord>( m, "EdgeIdAndCoord" ).
        def( pybind11::init<>() ).
        def_readwrite( "id", &EdgeIdAndCoord::id ).
        def_readwrite( "coord", &EdgeIdAndCoord::coord );

    pybind11::class_<TextMeshAlignParams>( m, "TextAlignParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "startPoint", &TextMeshAlignParams::startPoint ).
        def_readwrite( "direction", &TextMeshAlignParams::direction ).
        def_readwrite( "fontDetalization", &TextMeshAlignParams::fontDetalization ).
        def_readwrite( "text", &TextMeshAlignParams::text ).
        def_readwrite( "fontHeight ", &TextMeshAlignParams::fontHeight ).
        def_readwrite( "surfaceOffset", &TextMeshAlignParams::surfaceOffset ).
        def_readwrite( "textMaximumMovement", &TextMeshAlignParams::textMaximumMovement ).
        def_readwrite( "symbolsDistanceAdditionalOffset", &TextMeshAlignParams::symbolsDistanceAdditionalOffset ).
        def_readwrite( "pathToFontFile", &TextMeshAlignParams::pathToFontFile );
} )

Mesh createTextOnMesh( Mesh& mesh, const AffineXf3f& xf, TextMeshAlignParams params )
{
    if(params.pathToFontFile.empty())
        params.pathToFontFile = GetFontsDirectory().append( "Karla-Regular.ttf" );
    auto res = alignTextToMesh( mesh, xf, params );
    if(res)
        return res.value();
    else
        // failed to align
        return {};
}

MR_ADD_PYTHON_FUNCTION( mrmeshpy, create_text_on_mesh, &createTextOnMesh, "create text on mesh" )

// Laplacian Brush
// nothing to add or test

// Fix Undercuts
MR_ADD_PYTHON_FUNCTION( mrmeshpy, fix_undercuts_on_area,
    static_cast<void ( * )(Mesh&, const FaceBitSet&, const Vector3f&, float, float)> (&MR::FixUndercuts::fixUndercuts),
    "fill all undercuts in direction for fixed area" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, fix_undercuts,
    static_cast<void ( * )(Mesh&, const Vector3f&, float, float)> (&MR::FixUndercuts::fixUndercuts),
    "fill all undercuts in direction" )

// Free Form Transform
// nothing to add or test

// Collision Plugin
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, FaceFace, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::FaceFace>( m, "FaceFace" ).
        def( pybind11::init<>() ).
        def_readwrite( "aFace", &MR::FaceFace::aFace ).
        def_readwrite( "bFace", &MR::FaceFace::bFace );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorFaceFace, MR::FaceFace )

// TODO: introduce MeshPart
std::vector<FaceFace> findCollidingTrianglesSimple( const Mesh& a, const Mesh& b, const AffineXf3f * rigidB2A, bool firstIntersectionOnly )
{
    return findCollidingTriangles(a, b, rigidB2A, firstIntersectionOnly);
}
MR_ADD_PYTHON_FUNCTION( mrmeshpy, find_colliding_faces, &findCollidingTrianglesSimple, "finds all colliding face pairs" )

// Signed Distance
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshSignedDistanceResult, [] ( pybind11::module_& m )
{
    pybind11::class_<MeshSignedDistanceResult>( m, "MeshSignedDistanceResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "a", &MeshSignedDistanceResult::a ).
        def_readwrite( "b", &MeshSignedDistanceResult::b ).
        def_readwrite( "signedDist", &MeshSignedDistanceResult::signedDist );
} )

MeshSignedDistanceResult findSignedDistanceSimple( const Mesh& a, const Mesh& b, const AffineXf3f& rigidB2A )
{
    return findSignedDistance( a, b, &rigidB2A, std::numeric_limits<float>::max() );
}
MR_ADD_PYTHON_FUNCTION( mrmeshpy, find_signed_distance, &findSignedDistanceSimple, "finds signed distance for the current mesh. Negative value is for inner points" )

// Fix Spikes
MR_ADD_PYTHON_FUNCTION( mrmeshpy, remove_spikes, &removeSpikes, "removes spikes for the current mesh with given iterations" )

// Surface Distance
// nothing to add or test

// Geodesic Path
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, TriPoint, [] ( pybind11::module_& m )
{
    pybind11::class_<TriPointf>( m, "TriPoint" ).
        def( pybind11::init<>() ).
        def_readwrite( "a", &TriPointf::a ).
        def_readwrite( "b", &TriPointf::b );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshTriPoint, [] ( pybind11::module_& m )
{
    pybind11::class_<MeshTriPoint>( m, "MeshTriPoint" ).
        def( pybind11::init<>() ).
        def_readwrite( "e", &MeshTriPoint::e ).
        def_readwrite( "bary", &MeshTriPoint::bary );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshEdgePoint, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshEdgePoint>( m, "MeshEdgePoint" ).
        def( pybind11::init<>() ).
        def_readwrite( "e", &MR::MeshEdgePoint::e ).
        def_readwrite( "a", &MR::MeshEdgePoint::a );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorMeshEdgePoint, MR::MeshEdgePoint )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, compute_surface_path, &computeSurfacePath, "finds closest surface path between points" )

// Relax Mesh
MR_ADD_PYTHON_FUNCTION( mrmeshpy, relax, &relax, "relax mesh" )

// Re-mesh
MR_ADD_PYTHON_FUNCTION( mrmeshpy, make_delone_edge_flips, &makeDeloneEdgeFlips, "Delone flips edges" )

// Subdivider Plugin
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SubdivideSettings, [] ( pybind11::module_& m )
{
    pybind11::class_<SubdivideSettings>( m, "SubdivideSettings" ).
        def( pybind11::init<>() ).
        def_readwrite( "maxEdgeLen", &SubdivideSettings::maxEdgeLen ).
        def_readwrite( "maxEdgeSplits", &SubdivideSettings::maxEdgeSplits ).
        def_readwrite( "maxDeviationAfterFlip", &SubdivideSettings::maxDeviationAfterFlip );
} )

// TODO: introduce MeshPart
MR_ADD_PYTHON_FUNCTION( mrmeshpy, subdivide_mesh, &subdivideMesh, "split edges in mesh with settings" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, delete_faces, &deleteFaces, "delete faces from topology" )

// Distance Map
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, DistanceMap, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::DistanceMap>( m, "DistanceMap" ).
        def( pybind11::init<>() ).
        def( "get", static_cast< std::optional<float>( MR::DistanceMap::* )( size_t, size_t ) const >( &MR::DistanceMap::get ), "read X,Y value" ).
        def( "get", static_cast< std::optional<float>( MR::DistanceMap::* )( size_t ) const >( &MR::DistanceMap::get ), "read value by index" ).
        def( "getInterpolated", ( std::optional<float>( MR::DistanceMap::* )( float, float ) const )& MR::DistanceMap::getInterpolated, "bilinear interpolation between 4 pixels" ).
        def( "isValid", ( bool( MR::DistanceMap::* )( size_t, size_t ) const )& MR::DistanceMap::isValid, "check if X,Y pixel is valid" ).
        def( "isValid", ( bool( MR::DistanceMap::* )( size_t ) const )& MR::DistanceMap::isValid, "check if index pixel is valid").
        def( "resX", &MR::DistanceMap::resX, "X resolution" ).
        def( "resY", &MR::DistanceMap::resY, "Y resolution" ).
        def( "clear", &MR::DistanceMap::clear, "clear all values, set resolutions to zero" ).
        def( "invalidateAll", &MR::DistanceMap::invalidateAll, "invalidate all pixels" ).
        def( "set", static_cast< void( MR::DistanceMap::* )( size_t, float ) >( &MR::DistanceMap::set ), "write value by index" ).
        def( "set", static_cast< void( MR::DistanceMap::* )( size_t, size_t, float ) >( &MR::DistanceMap::set ), "write X,Y value" ).
        def( "unset", static_cast< void( MR::DistanceMap::* )( size_t, size_t ) >( &MR::DistanceMap::unset), "invalidate X,Y pixel" ).
        def( "unset", static_cast< void( MR::DistanceMap::* )( size_t ) >( &MR::DistanceMap::unset), "invalidate by index" );
} )

// MeshToDistanceMapParams
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshToDistanceMapParams, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshToDistanceMapParams>( m, "MeshToDistanceMapParams" ).
        def( pybind11::init<>() ).
        def( "setDistanceLimits", &MR::MeshToDistanceMapParams::setDistanceLimits ).
        def_readwrite( "xRange", &MR::MeshToDistanceMapParams::xRange ).
        def_readwrite( "yRange", &MR::MeshToDistanceMapParams::yRange ).
        def_readwrite( "direction", &MR::MeshToDistanceMapParams::direction ).
        def_readwrite( "orgPoint", &MR::MeshToDistanceMapParams::orgPoint ).
        def_readwrite( "useDistanceLimits", &MR::MeshToDistanceMapParams::useDistanceLimits ).
        def_readwrite( "allowNegativeValues", &MR::MeshToDistanceMapParams::allowNegativeValues ).
        def_readwrite( "minValue", &MR::MeshToDistanceMapParams::minValue ).
        def_readwrite( "maxValue", &MR::MeshToDistanceMapParams::maxValue ).
        def_readwrite( "resolution", &MR::MeshToDistanceMapParams::resolution );
} )

// Subdivider Plugin

MR::DistanceMap computeDMwithParams( const Mesh& mesh, const MeshToDistanceMapParams& params )
{
    return computeDistanceMapD( mesh, params );
}

MR_ADD_PYTHON_FUNCTION( mrmeshpy, compute_distance_map,
    &computeDMwithParams,
    "computes Distance Map with given xf transformation, pixel size. Precise bounding box computation as the last parameter" )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, get_distance_map_mesh, &distanceMapToMesh,
    "converts distance map to mesh" )

// TODO: introduce filesystem::path
void saveDistanceMapToImageSimple(const DistanceMap& dm, const std::string& filename)
{
    saveDistanceMapToImage(dm, filename);
}
MR_ADD_PYTHON_FUNCTION( mrmeshpy, save_depth_image, &saveDistanceMapToImageSimple, "saves distance map to image file" )

// Position Verts Smooth
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LaplacianEdgeWeightsParam, [] ( pybind11::module_& m )
{
    pybind11::enum_<Laplacian::EdgeWeights>( m, "LaplacianEdgeWeightsParam" ).
        value( "Unit", Laplacian::EdgeWeights::Unit ).
        value( "Cotan", Laplacian::EdgeWeights::Cotan );
} )

MR_ADD_PYTHON_FUNCTION( mrmeshpy, position_verts_smoothly, &positionVertsSmoothly, "shifts vertices to make smooth surface by Unit Laplacian" )
