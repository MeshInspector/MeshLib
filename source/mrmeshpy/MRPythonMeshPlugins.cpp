#include "MRMesh/MRPython.h"
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
#include "MRMesh/MRMeshFixer.h"
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

// Boolean
Mesh booleanSub( const Mesh& mesh1, const Mesh& mesh2, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    auto gridB = convert(mesh2);
    gridA -= gridB;
    return convert(gridA);
}

Mesh booleanUnion( const Mesh& mesh1, const Mesh& mesh2, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    auto gridB = convert(mesh2);
    gridA += gridB;
    return convert( gridA );
}

Mesh booleanIntersect( const Mesh& mesh1, const Mesh& mesh2, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    auto gridB = convert(mesh2);
    gridA *= gridB;
    return convert( gridA );
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, VoxelBooleanBlock, [] ( pybind11::module_& m )
{
    m.def( "fixSelfIntersections", &fixSelfIntersections, pybind11::arg( "mesh" ), pybind11::arg( "voxelSize" ), "fix self-intersections by converting to voxels and back" );
    m.def( "voxelBooleanSubtract", &booleanSub, pybind11::arg( "meshA" ), pybind11::arg( "meshB" ), pybind11::arg( "voxelSize" ), "subtract mesh B from mesh A" );
    m.def( "voxelBooleanUnite", &booleanUnion, pybind11::arg( "meshA" ), pybind11::arg( "meshB" ), pybind11::arg( "voxelSize" ), "unite mesh A and mesh B" );
    m.def( "voxelBooleanIntersect", &booleanIntersect, pybind11::arg( "meshA" ), pybind11::arg( "meshB" ), pybind11::arg( "voxelSize" ), "intersect mesh A and mesh B" );

} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, DegenerationsDetection, [] ( pybind11::module_& m )
{
    m.def( "detectTunnelFaces", &MR::detectTunnelFaces, pybind11::arg( "mp" ), pybind11::arg( "maxTunnelLength" ), 
        "returns tunnels as a number of faces;\n"
        "if you remove these faces and patch every boundary with disk, then the surface will be topology equivalent to sphere" );

    m.def( "detectBasisTunnels", &MR::detectBasisTunnels, pybind11::arg( "mp" ), "detects all not-contractible-in-point and not-equivalent tunnel loops on the mesh" );

    m.def( "findDegenerateFaces", &MR::findDegenerateFaces, 
        pybind11::arg( "mesh" ), pybind11::arg( "criticalAspectRatio" ) = FLT_MAX,
        "finds faces which aspect ratio >= criticalAspectRatio" );

    m.def( "fixMultipleEdges", ( void( * )( MR::Mesh& ) )& MR::fixMultipleEdges,
        pybind11::arg( "mesh" ), "finds and resolves multiple edges" );

    m.def( "hasMultipleEdges", &MR::hasMultipleEdges,
        pybind11::arg( "topology" ), "finds multiple edges in the mesh" );

    m.def( "removeSpikes", &MR::removeSpikes,
        pybind11::arg( "mesh" ), pybind11::arg( "maxIterations" ), pybind11::arg( "minSumAngle" ), pybind11::arg( "region" ) = nullptr, 
        "applies at most given number of relaxation iterations the spikes detected by given threshold" );

    m.def( "fixUndercuts", ( void ( * )( Mesh&, const FaceBitSet&, const Vector3f&, float, float ) )& MR::FixUndercuts::fixUndercuts,
        pybind11::arg( "mesh" ), pybind11::arg( "selectedArea" ), pybind11::arg( "upDirection" ), pybind11::arg( "voxelSize" ) = 0.0f, pybind11::arg( "bottomExtension" ) = 0.0f,
        "aChanges mesh:\n"
        "Fills all holes first, then:\n"
        "fixes undercuts (in selected area) via prolonging widest points down\n"
        "Requires to update RenderObject after using\n"
        "upDirection is in mesh space\n"
        "voxelSize -  size of voxel in mesh rasterization, precision grows with lower voxelSize\n"
        "bottomExtension - this parameter specifies how long should bottom prolongation be, if (bottomExtension <= 0) bottomExtension = 2*voxelSize\n"
        "\tif mesh is not closed this is used to prolong hole and make bottom\n"
        "\nif voxelSize == 0.0f it will be counted automaticly" );

    m.def( "fixUndercuts", ( void ( * )( Mesh&, const Vector3f&, float, float ) )& MR::FixUndercuts::fixUndercuts,
    pybind11::arg( "mesh" ), pybind11::arg( "upDirection" ), pybind11::arg( "voxelSize" ) = 0.0f, pybind11::arg( "bottomExtension" ) = 0.0f,
        "aChanges mesh:\n"
        "Fills all holes first, then:\n"
        "fixes undercuts via prolonging widest points down\n"
        "Requires to update RenderObject after using\n"
        "upDirection is in mesh space\n"
        "voxelSize -  size of voxel in mesh rasterization, precision grows with lower voxelSize\n"
        "bottomExtension - this parameter specifies how long should bottom prolongation be, if (bottomExtension <= 0) bottomExtension = 2*voxelSize\n"
        "\tif mesh is not closed this is used to prolong hole and make bottom\n"
        "\nif voxelSize == 0.0f it will be counted automaticly" );
} )


Mesh createTextOnMesh( Mesh& mesh, const AffineXf3f& xf, TextMeshAlignParams params )
{
    if ( params.pathToFontFile.empty() )
        params.pathToFontFile = GetFontsDirectory().append( "Karla-Regular.ttf" );
    auto res = alignTextToMesh( mesh, xf, params );
    if ( res )
        return res.value();
    else
        // failed to align
        return {};
}

// Text Mesh
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SymbolMeshParams, [] ( pybind11::module_& m )
{
    pybind11::class_<SymbolMeshParams>( m, "SymbolMeshParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "text", &SymbolMeshParams::text, "Text that will be made mesh" ).
        def_readwrite( "fontDetalization", &SymbolMeshParams::fontDetalization, "Detailization of Bezier curves on font glyphs" ).
        def_readwrite( "symbolsDistanceAdditionalOffset", &SymbolMeshParams::symbolsDistanceAdditionalOffset,
            "Additional offset between symbols (in symbol size: 1.0f adds one \"space\", 0.5 adds half \"space\")\n"
            "should be >= 0.0f" ).
        def_readwrite( "pathToFontFile", &TextMeshAlignParams::pathToFontFile, "Path to font file" );

    m.def( "createSymbolsMesh", &MR::createSymbolsMesh, pybind11::arg( "params" ), "converts text string into Z-facing symbol mesh" );

    pybind11::class_<EdgeIdAndCoord>( m, "EdgeIdAndCoord", "This structure represents point on mesh, by EdgeId (point should be in left triangle of this edge) and coordinate" ).
        def( pybind11::init<>() ).
        def_readwrite( "id", &EdgeIdAndCoord::id ).
        def_readwrite( "coord", &EdgeIdAndCoord::coord );

    pybind11::class_<TextMeshAlignParams, SymbolMeshParams>( m, "TextAlignParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "startPoint", &TextMeshAlignParams::startPoint, "Start coordinate on mesh, represent lowest left corner of text" ).
        def_readwrite( "direction", &TextMeshAlignParams::direction, "Direction of text" ).
        def_readwrite( "fontHeight ", &TextMeshAlignParams::fontHeight, "Font height, meters" ).
        def_readwrite( "surfaceOffset", &TextMeshAlignParams::surfaceOffset, "Text mesh inside and outside offset of input mesh" ).
        def_readwrite( "textMaximumMovement", &TextMeshAlignParams::textMaximumMovement, "Maximum possible movement of text mesh alignment, meters" );

    m.def( "alignTextToMesh", &createTextOnMesh, pybind11::arg( "mesh" ), pybind11::arg( "xf" ), pybind11::arg( "params" ),
        "create text on mesh" );
} )


MR_ADD_PYTHON_VEC( mrmeshpy, vectorMeshEdgePoint, MR::MeshEdgePoint )

// Signed Distance
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshSignedDistanceResult, [] ( pybind11::module_& m )
{
    pybind11::class_<MeshSignedDistanceResult>( m, "MeshSignedDistanceResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "a", &MeshSignedDistanceResult::a, "two closest points: from meshes A and B respectively" ).
        def_readwrite( "b", &MeshSignedDistanceResult::b, "two closest points: from meshes A and B respectively" ).
        def_readwrite( "signedDist", &MeshSignedDistanceResult::signedDist, "signed distance between a and b, positive if meshes do not collide" );

    m.def( "findSignedDistance", ( MeshSignedDistanceResult( * )( const MeshPart&, const MeshPart&, const AffineXf3f*, float ) )& MR::findSignedDistance,
        pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "upDistLimitSq" ) = FLT_MAX,
        "computes minimal distance between two meshes\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation\n"
        "\tupDistLimitSq - upper limit on the positive distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid points" );

    m.def("findMaxDistanceSqOneWay",&MR::findMaxDistanceSqOneWay, 
        pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "upDistLimitSq" ) = FLT_MAX,
        "returns the maximum of the distances from each B-mesh point to A-mesh\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation\n"
        "\tmaxDistanceSq - upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq" );

    m.def( "findMaxDistanceSq", &MR::findMaxDistanceSq,
        pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "upDistLimitSq" ) = FLT_MAX,
        "returns the maximum of the distances from each mesh point to another mesh in both directions\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation\n"
        "\tmaxDistanceSq - upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq" );
} )

// Relax Mesh
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Relax, [] ( pybind11::module_& m )
{
    pybind11::class_<RelaxParams>( m, "RelaxParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "force", &RelaxParams::force, "speed of relaxing, typical values (0.0, 0.5]" ).
        def_readwrite( "iterations", &RelaxParams::iterations, "number of iterations" ).
        def_readwrite( "region", &RelaxParams::region, "region to relax" );

    pybind11::class_<MeshRelaxParams, RelaxParams>( m, "MeshRelaxParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "hardSmoothTetrahedrons", &MeshRelaxParams::hardSmoothTetrahedrons, "smooth tetrahedron verts (with complete three edges ring) to base triangle (based on its edges destinations)" );

    m.def( "relax", [] ( Mesh& mesh, const MeshRelaxParams& params )
    {
        return relax( mesh, params ); // lambda to skip progress callback parameter
    },
        pybind11::arg( "mesh" ), pybind11::arg( "params" ) = MeshRelaxParams{},
        "applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )\n"
        "return true if was finished successfully, false if was interrupted by progress callback");
} )

// Subdivider Plugin
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SubdivideSettings, [] ( pybind11::module_& m )
{
    pybind11::class_<SubdivideSettings>( m, "SubdivideSettings" ).
        def( pybind11::init<>() ).
        def_readwrite( "maxEdgeLen", &SubdivideSettings::maxEdgeLen, "Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value" ).
        def_readwrite( "maxEdgeSplits", &SubdivideSettings::maxEdgeSplits, "Maximum number of edge splits allowed" ).
        def_readwrite( "maxDeviationAfterFlip", &SubdivideSettings::maxDeviationAfterFlip, "Improves local mesh triangulation by doing edge flips if it does not make too big surface deviation" ).
        def_readwrite( "maxAngleChangeAfterFlip", &SubdivideSettings::maxAngleChangeAfterFlip, "Improves local mesh triangulation by doing edge flips if it does change dihedral angle more than on this value" ).
        def_readwrite( "region ", &SubdivideSettings::region, "Region on mesh to be subdivided, it is updated during the operation" ).
        def_readwrite( "newVerts ", &SubdivideSettings::newVerts, "New vertices appeared during subdivision will be added here" ).
        def_readwrite( "subdivideBorder", &SubdivideSettings::subdivideBorder,
            "If false do not touch border edges (cannot subdivide lone faces)\n"
            "use findRegionOuterFaces to find boundary faces" ).
        def_readwrite( "critAspectRatio", &SubdivideSettings::critAspectRatio,
            "If subdivideBorder is off subdivider can produce narrow triangles near border\n"
            "this parameter prevents subdivision of such triangles" ).
        def_readwrite( "useCurvature", &SubdivideSettings::useCurvature,
            "This option works best for natural surfaces, where all triangles are close to equilateral and have similar area,\n"
            "and no sharp edges in between" ).
        def_readwrite( "newVerts ", &SubdivideSettings::newVerts, "New vertices appeared during subdivision will be added here" );

    m.def( "subdivideMesh", &MR::subdivideMesh,
        pybind11::arg( "mesh" ), pybind11::arg( "settings" ) = MR::SubdivideSettings{},
        "Split edges in mesh region according to the settings;\n"
        "return The total number of edge splits performed" );

} )


void saveDistanceMapToImageSimple( const DistanceMap& dm, const std::string& filename, float trashold )
{
    saveDistanceMapToImage( dm, filename, trashold );
}

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


    pybind11::class_<MR::MeshToDistanceMapParams>( m, "MeshToDistanceMapParams" ).
        def( pybind11::init<>(), "default constructor. Manual params initialization is required" ).
        def( "setDistanceLimits", &MR::MeshToDistanceMapParams::setDistanceLimits, pybind11::arg( "min" ), pybind11::arg( "max" ),
            "if distance is not in set range, pixel became invalid\n"
            "default value: false. Any distance will be applied (include negative)" ).
        def_readwrite( "xRange", &MR::MeshToDistanceMapParams::xRange, "Cartesian range vector between distance map borders in X direction" ).
        def_readwrite( "yRange", &MR::MeshToDistanceMapParams::yRange, "Cartesian range vector between distance map borders in Y direction" ).
        def_readwrite( "direction", &MR::MeshToDistanceMapParams::direction, "direction of intersection ray" ).
        def_readwrite( "orgPoint", &MR::MeshToDistanceMapParams::orgPoint, "location of (0,0) pixel with value 0.f" ).
        def_readwrite( "useDistanceLimits", &MR::MeshToDistanceMapParams::useDistanceLimits, "out of limits intersections will be set to non-valid" ).
        def_readwrite( "allowNegativeValues", &MR::MeshToDistanceMapParams::allowNegativeValues, "allows to find intersections in backward to direction vector with negative values" ).
        def_readwrite( "minValue", &MR::MeshToDistanceMapParams::minValue, "Using of this parameter depends on useDistanceLimits" ).
        def_readwrite( "maxValue", &MR::MeshToDistanceMapParams::maxValue, "Using of this parameter depends on useDistanceLimits" ).
        def_readwrite( "resolution", &MR::MeshToDistanceMapParams::resolution, "resolution of distance map" );

    m.def( "computeDistanceMapD", &MR::computeDistanceMapD, pybind11::arg( "mp" ), pybind11::arg( "params" ),
        "computes distance map for presented projection parameters\n"
        "use MeshToDistanceMapParams constructor instead of overloads of this function\n"
        "MeshPart - input 3d model\n"
        "general call. You could customize params manually" );

    m.def( "saveDistanceMapToImage", &saveDistanceMapToImageSimple, 
        pybind11::arg( "distMap" ), pybind11::arg( "filename" ), pybind11::arg( "threshold" ) = 1.0f / 255.0f,
        "saves distance map to monochrome image in scales of gray:\n"
        "\tthreshold - threshold of maximum values [0.; 1.]. invalid pixel set as 0. (black)\n"
        "minimum (close): 1.0 (white)\n"
        "maximum (far): threshold\n"
        "invalid (infinity): 0.0 (black)" );
} )

// Position Verts Smooth
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LaplacianEdgeWeightsParam, [] ( pybind11::module_& m )
{
    pybind11::enum_<Laplacian::EdgeWeights>( m, "LaplacianEdgeWeightsParam" ).
        value( "Unit", Laplacian::EdgeWeights::Unit, "all edges have same weight=1" ).
        value( "Cotan", Laplacian::EdgeWeights::Cotan, "edge weight depends on local geometry and uses cotangent values" ).
        value( "CotanTimesLength", Laplacian::EdgeWeights::CotanTimesLength, "edge weight is equal to edge length times cotangent weight" );

    m.def( "positionVertsSmoothly", &MR::positionVertsSmoothly,
        pybind11::arg( "mesh" ), pybind11::arg( "verts" ), pybind11::arg( "egdeWeightsType" ) = MR::Laplacian::EdgeWeights::Cotan,
        "Puts given vertices in such positions to make smooth surface both inside verts-region and on its boundary" );
} )
