#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MROffset.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRFloatGrid.h"
#include "MRVDBConversions.h"
#include "MRTimer.h"
#include "MRPolyline.h"
#include "MRMeshFillHole.h"
#include "MRRegionBoundary.h"
#include "MRPch/MRSpdlog.h"
#include "MRVoxelsConversions.h"
#include "MRSharpenMarchingCubesMesh.h"
#include "MRFastWindingNumber.h"
#include "MRVolumeIndexer.h"
#include <thread>

namespace
{
constexpr float autoVoxelNumber = 5e6f;
}

namespace MR
{

tl::expected<Mesh, std::string> offsetMesh( const MeshPart & mp, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER

    float voxelSize = params.voxelSize;
    // Compute voxel size if needed
    if ( voxelSize <= 0.0f )
    {
        auto bb = mp.mesh.computeBoundingBox( mp.region );
        auto vol = bb.volume();
        voxelSize = std::cbrt( vol / autoVoxelNumber );
    }

    bool useShell = params.type == OffsetParameters::Type::Shell;
    bool signPostprocess = !findLeftBoundary( mp.mesh.topology, mp.region ).empty() && !useShell;

    if ( useShell )
        offset = std::abs( offset );

    auto offsetInVoxels = offset / voxelSize;

    auto voxelSizeVector = Vector3f::diagonal( voxelSize );
    // Make grid
    FloatGrid grid;
    if ( !useShell && !signPostprocess ) 
    {
        // Compute signed distance grid
        grid = meshToLevelSet( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 2,
            subprogress( params.callBack, 0.0f, signPostprocess ? 0.33f : 0.5f ) );
    }
    else
    {
        // Compute unsigned distance grid
        grid = meshToDistanceField( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 2,
                        subprogress( params.callBack, 0.0f, signPostprocess ? 0.33f : 0.5f ) );
        if ( grid ) // to flip mesh normals
            grid->setGridClass( openvdb::GRID_LEVEL_SET );
    }

    if ( !grid )
        return tl::make_unexpected( "Operation was canceled." );

    if ( signPostprocess )
    {
        // Compute signs for initially unsigned distance field
        auto sp = subprogress( params.callBack, 0.33f, 0.66f );
        auto signRes = makeSignedWithFastWinding( grid, Vector3f::diagonal( voxelSize ), mp.mesh, {}, sp );
        if ( !signRes.has_value() )
            return tl::make_unexpected( signRes.error() );
    }

    // Make offset mesh
    auto newMesh = gridToMesh( std::move( grid ), voxelSizeVector, offsetInVoxels, params.adaptivity, 
        subprogress( params.callBack, signPostprocess ? 0.66f : 0.5f, 1.0f ) );

    if ( !newMesh.has_value() )
        return tl::make_unexpected( "Operation was canceled." );

    return newMesh;
}

tl::expected<Mesh, std::string> doubleOffsetMesh( const MeshPart& mp, float offsetA, float offsetB, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER
    if ( params.type == OffsetParameters::Type::Shell )
    {
        spdlog::warn( "Cannot use shell for double offset, using offset mode instead." );
    }
    return levelSetDoubleConvertion( mp, AffineXf3f(), params.voxelSize, offsetA, offsetB, params.adaptivity, params.callBack );
}

tl::expected<Mesh, std::string> mcOffsetMesh( const Mesh& mesh, float offset, 
    const BaseOffsetParameters& params, Vector<VoxelId, FaceId> * outMap )
{
    MR_TIMER;
    auto meshToLSCb = subprogress( params.callBack, 0.0f, 0.4f );
    if ( !params.simpleVolumeSignMode )
    {
        auto offsetInVoxels = offset / params.voxelSize;
        auto voxelRes = meshToLevelSet( mesh, AffineXf3f(),
            Vector3f::diagonal( params.voxelSize ),
            std::abs( offsetInVoxels ) + 2, meshToLSCb );
        if ( !voxelRes )
            return tl::make_unexpected( "Operation was canceled." );

        VdbVolume volume = floatGridToVdbVolume( voxelRes );

        VolumeToMeshParams vmParams;
        vmParams.basis.A = Matrix3f::scale( params.voxelSize );
        vmParams.iso = offsetInVoxels;
        vmParams.lessInside = true;
        vmParams.cb = subprogress( params.callBack, 0.4f, 1.0f );
        vmParams.outVoxelPerFaceMap = outMap;
        auto meshRes = vdbVolumeToMesh( volume, vmParams );
        if ( !meshRes )
            return tl::make_unexpected( "Operation was canceled." );

        return std::move( *meshRes );
    }
    else
    {
        MeshToSimpleVolumeParams msParams;
        msParams.cb = meshToLSCb;
        auto box = mesh.getBoundingBox();
        auto absOffset = std::abs( offset );
        auto expansion = 3.0f * Vector3f::diagonal( params.voxelSize ) + 2.0f * Vector3f::diagonal( absOffset );
        msParams.basis.b = box.min - expansion;
        msParams.basis.A = Matrix3f::scale( params.voxelSize );
        msParams.dimensions = Vector3i( ( box.max + expansion - msParams.basis.b ) / params.voxelSize ) + Vector3i::diagonal( 1 );
        msParams.signMode = *params.simpleVolumeSignMode;
        msParams.maxDistSq = sqr( absOffset + params.voxelSize );
        msParams.minDistSq = sqr( std::max( absOffset - params.voxelSize, 0.0f ) );
        
        auto volume = meshToSimpleVolume( mesh, msParams );
        if ( !volume )
            return tl::make_unexpected( "Operation was canceled." );

        VolumeToMeshParams vmParams;
        vmParams.basis = msParams.basis;
        vmParams.iso = offset;
        vmParams.cb = subprogress( params.callBack, 0.4f, 1.0f );
        vmParams.lessInside = true;
        vmParams.outVoxelPerFaceMap = outMap;
        auto meshRes = simpleVolumeToMesh( std::move( *volume ), vmParams );
        if ( !meshRes )
            return tl::make_unexpected( "Operation was canceled." );
        return std::move( *meshRes );
    }
}

tl::expected<MR::Mesh, std::string> sharpOffsetMesh( const Mesh& mesh, float offset, const SharpOffsetParameters& params )
{
    MR_TIMER
    BaseOffsetParameters mcParams = params;
    mcParams.callBack = subprogress( params.callBack, 0.0f, 0.7f );
    Vector<VoxelId, FaceId> map;
    auto res = mcOffsetMesh( mesh, offset, mcParams, &map );
    if ( !res.has_value() )
        return res;

    SharpenMarchingCubesMeshSettings sharpenParams;
    sharpenParams.minNewVertDev = params.voxelSize * params.minNewVertDev;
    sharpenParams.maxNewRank2VertDev = params.voxelSize * params.maxNewRank2VertDev;
    sharpenParams.maxNewRank3VertDev = params.voxelSize * params.maxNewRank3VertDev;
    sharpenParams.maxOldVertPosCorrection = params.voxelSize * params.maxOldVertPosCorrection;
    sharpenParams.offset = offset;
    sharpenParams.outSharpEdges = params.outSharpEdges;

    sharpenMarchingCubesMesh( mesh, res.value(), map, sharpenParams );
    if ( !reportProgress( params.callBack, 0.99f ) )
        return tl::make_unexpected( "Operation was canceled." );

    return res;
}

tl::expected<Mesh, std::string> offsetPolyline( const Polyline3& polyline, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER;

    Mesh mesh;
    auto contours = polyline.topology.convertToContours<Vector3f>(
        [&points = polyline.points]( VertId v )
    {
        return points[v];
    } );

    std::vector<EdgeId> newHoles;
    newHoles.reserve( contours.size() );
    for ( auto& cont : contours )
    {
        if ( cont[0] != cont.back() )
            cont.insert( cont.end(), cont.rbegin(), cont.rend() );
        newHoles.push_back( mesh.addSeparateEdgeLoop( cont ) );
    }

    for ( auto h : newHoles )
        makeDegenerateBandAroundHole( mesh, h );

    return offsetMesh( mesh, offset, params );
}

}
#endif
