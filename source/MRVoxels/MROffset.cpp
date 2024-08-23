#include "MROffset.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBox.h"
#include "MRFloatGrid.h"
#include "MRVDBConversions.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRMeshFillHole.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRVoxelsConversions.h"
#include "MRMesh/MRSharpenMarchingCubesMesh.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRInnerShell.h"
#include "MRMesh/MRMeshFixer.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRRingIterator.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

float suggestVoxelSize( const MeshPart & mp, float approxNumVoxels )
{
    MR_TIMER
    auto bb = mp.mesh.computeBoundingBox( mp.region );
    auto vol = bb.volume();
    return std::cbrt( vol / approxNumVoxels );
}

Expected<Mesh> offsetMesh( const MeshPart & mp, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER
    assert( params.signDetectionMode == SignDetectionMode::Unsigned 
        || params.signDetectionMode == SignDetectionMode::OpenVDB
        || params.signDetectionMode == SignDetectionMode::HoleWindingRule );

    if ( params.voxelSize <= 0 )
    {
        assert( false );
        return unexpected( "wrong voxelSize" );
    }

    float voxelSize = params.voxelSize;
    bool useShell = params.signDetectionMode == SignDetectionMode::Unsigned;
    bool signPostprocess = params.signDetectionMode == SignDetectionMode::HoleWindingRule;

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
            subprogress( params.callBack, 0.0f, 0.5f ) );
    }
    else
    {
        // Compute unsigned distance grid
        grid = meshToDistanceField( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 2,
                        subprogress( params.callBack, 0.0f, signPostprocess ? 0.33f : 0.5f ) );
        setLevelSetType( grid ); // to flip mesh normals
    }

    if ( !grid )
        return unexpectedOperationCanceled();

    if ( signPostprocess )
    {
        auto signRes = makeSignedByWindingNumber( grid, Vector3f::diagonal( voxelSize ), mp.mesh,
        {
            .fwn = params.fwn,
            .windingNumberThreshold = params.windingNumberThreshold,
            .windingNumberBeta = params.windingNumberBeta,
            .progress = subprogress( params.callBack, 0.33f, 0.66f )
        } );
        if ( !signRes.has_value() )
            return unexpected( std::move( signRes.error() ) );
    }

    // Make offset mesh
    auto newMesh = gridToMesh( std::move( grid ), GridToMeshSettings{
        .voxelSize = voxelSizeVector,
        .isoValue = offsetInVoxels,
        .adaptivity = 0, // it does not work good, better use common decimation after offsetting
        .cb = subprogress( params.callBack, signPostprocess ? 0.66f : 0.5f, 1.0f )
    } );

    if ( !newMesh.has_value() )
        return unexpectedOperationCanceled();

    return newMesh;
}

Expected<Mesh> doubleOffsetMesh( const MeshPart& mp, float offsetA, float offsetB, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER
    if ( params.signDetectionMode == SignDetectionMode::Unsigned )
    {
        spdlog::warn( "Cannot use shell for double offset, using offset mode instead." );
    }
    return doubleOffsetVdb( mp,
    {
        .voxelSize = params.voxelSize,
        .offsetA = offsetA,
        .offsetB = offsetB,
        .fwn = params.fwn,
        .windingNumberThreshold = params.windingNumberThreshold,
        .windingNumberBeta = params.windingNumberBeta,
        .progress = params.callBack
    } );
}

Expected<Mesh> mcOffsetMesh( const MeshPart& mp, float offset,
    const OffsetParameters& params, Vector<VoxelId, FaceId> * outMap )
{
    MR_TIMER;
    auto meshToLSCb = subprogress( params.callBack, 0.0f, 0.4f );
    if ( params.signDetectionMode == SignDetectionMode::OpenVDB )
    {
        auto offsetInVoxels = offset / params.voxelSize;
        auto voxelRes = meshToLevelSet( mp, AffineXf3f(),
            Vector3f::diagonal( params.voxelSize ),
            std::abs( offsetInVoxels ) + 2, meshToLSCb );
        if ( !voxelRes )
            return unexpectedOperationCanceled();

        VdbVolume volume = floatGridToVdbVolume( std::move( voxelRes ) );
        volume.voxelSize = Vector3f::diagonal( params.voxelSize );

        MarchingCubesParams vmParams;
        vmParams.iso = offsetInVoxels;
        vmParams.lessInside = true;
        vmParams.cb = subprogress( params.callBack, 0.4f, 1.0f );
        vmParams.outVoxelPerFaceMap = outMap;
        vmParams.freeVolume = [&volume]
        {
            Timer t( "~FloatGrid" );
            volume.data.reset();
        };
        return marchingCubes( volume, vmParams );
    }
    else
    {
        const bool funcVolume = !params.fwn && params.memoryEfficient;
        MeshToDistanceVolumeParams msParams;
        if ( !funcVolume )
            msParams.vol.cb = meshToLSCb;
        auto box = mp.mesh.computeBoundingBox( mp.region );
        auto absOffset = std::abs( offset );
        auto expansion = Vector3f::diagonal( 2 * params.voxelSize + absOffset );
        msParams.vol.origin = box.min - expansion;
        msParams.vol.voxelSize = Vector3f::diagonal( params.voxelSize );
        msParams.vol.dimensions = Vector3i( ( box.max + expansion - msParams.vol.origin ) / params.voxelSize ) + Vector3i::diagonal( 1 );
        msParams.dist.maxDistSq = sqr( absOffset + params.voxelSize );
        msParams.dist.minDistSq = sqr( std::max( absOffset - params.voxelSize, 0.0f ) );
        msParams.dist.signMode = params.signDetectionMode;
        msParams.dist.windingNumberThreshold = params.windingNumberThreshold;
        msParams.dist.windingNumberBeta = params.windingNumberBeta;
        msParams.fwn = params.fwn;

        MarchingCubesParams vmParams;
        vmParams.origin = msParams.vol.origin;
        vmParams.iso = offset;
        vmParams.cb = funcVolume ? params.callBack : subprogress( params.callBack, 0.4f, 1.0f );
        vmParams.lessInside = true;
        vmParams.outVoxelPerFaceMap = outMap;

        if ( funcVolume )
        {
            return marchingCubes( meshToDistanceFunctionVolume( mp, msParams ), vmParams );
        }
        else
        {
            return meshToDistanceVolume( mp, msParams ).and_then( [&vmParams] ( SimpleVolume&& volume )
            {
                vmParams.freeVolume = [&volume]
                {
                    Timer t( "~SimpleVolume" );
                    volume = {};
                };
                return marchingCubes( volume, vmParams );
            } );
        }
    }
}

Expected<Mesh> mcShellMeshRegion( const Mesh& mesh, const FaceBitSet& region, float offset,
    const BaseShellParameters& params, Vector<VoxelId, FaceId> * outMap )
{
    MR_TIMER

    DistanceVolumeParams dvParams;
    dvParams.cb = subprogress( params.callBack, 0.0f, 0.5f );
    auto box = mesh.getBoundingBox();
    auto absOffset = std::abs( offset );
    auto expansion = Vector3f::diagonal( 2 * params.voxelSize + absOffset );
    dvParams.origin = box.min - expansion;
    dvParams.voxelSize = Vector3f::diagonal( params.voxelSize );
    dvParams.dimensions = Vector3i( ( box.max + expansion - dvParams.origin ) / params.voxelSize ) + Vector3i::diagonal( 1 );

    auto volume = meshRegionToIndicatorVolume( mesh, region, offset, dvParams );
    if ( !volume )
        return unexpectedOperationCanceled();

    MarchingCubesParams vmParams;
    vmParams.origin = dvParams.origin;
    vmParams.iso = 0;
    vmParams.cb = subprogress( params.callBack, 0.5f, 1.0f );
    vmParams.lessInside = true;
    vmParams.outVoxelPerFaceMap = outMap;
    vmParams.freeVolume = [&volume]
    {
        Timer t( "~SimpleVolume" );
        volume = {};
    };
    return marchingCubes( *volume, vmParams );
}

Expected<Mesh> sharpOffsetMesh( const MeshPart& mp, float offset, const SharpOffsetParameters& params )
{
    MR_TIMER
    OffsetParameters mcParams = params;
    mcParams.callBack = subprogress( params.callBack, 0.0f, 0.7f );
    Vector<VoxelId, FaceId> map;
    auto res = mcOffsetMesh( mp, offset, mcParams, &map );
    if ( !res.has_value() )
        return res;

    SharpenMarchingCubesMeshSettings sharpenParams;
    sharpenParams.minNewVertDev = params.voxelSize * params.minNewVertDev;
    sharpenParams.maxNewRank2VertDev = params.voxelSize * params.maxNewRank2VertDev;
    sharpenParams.maxNewRank3VertDev = params.voxelSize * params.maxNewRank3VertDev;
    sharpenParams.maxOldVertPosCorrection = params.voxelSize * params.maxOldVertPosCorrection;
    sharpenParams.offset = offset;
    sharpenParams.outSharpEdges = params.outSharpEdges;

    sharpenMarchingCubesMesh( mp, res.value(), map, sharpenParams );
    if ( !reportProgress( params.callBack, 0.99f ) )
        return unexpectedOperationCanceled();

    return res;
}

Expected<Mesh> generalOffsetMesh( const MeshPart& mp, float offset, const GeneralOffsetParameters& params )
{
    switch( params.mode )
    {
    default:
        assert( false );
        [[fallthrough]];
    case GeneralOffsetParameters::Mode::Smooth:
        return offsetMesh( mp, offset, params );
    case GeneralOffsetParameters::Mode::Standard:
        return mcOffsetMesh( mp, offset, params );
    case GeneralOffsetParameters::Mode::Sharpening:
        return sharpOffsetMesh( mp, offset, params );
    }
}

Expected<Mesh> thickenMesh( const Mesh& mesh, float offset, const GeneralOffsetParameters& params )
{
    MR_TIMER
    const bool unsignedOffset = params.signDetectionMode == SignDetectionMode::Unsigned;
    auto res = generalOffsetMesh( mesh, unsignedOffset ? std::abs( offset ) : offset, params );
    if ( !res )
        return res;

    auto & resMesh = res.value();

    if ( unsignedOffset )
    {
        // delete shell faces from resMesh that project on wrong side of input mesh

        // do not trust degenerate faces with huge aspect ratios
        auto badFaces = findDegenerateFaces( mesh, 1000 ).value();
        // do not trust only boundary degenerate faces (excluding touching the boundary only by short edge)
        BitSetParallelFor( badFaces, [&] ( FaceId f )
        {
            float perimeter = 0;
            float bdLen = 0;
            for ( EdgeId e : leftRing( mesh.topology, f ) )
            {
                auto elen = mesh.edgeLength( e );
                perimeter += elen;
                if ( mesh.topology.isBdEdge( e ) )
                    bdLen += elen;
            }
            if ( perimeter * 0.1f >= bdLen )
                badFaces.reset( f );
        } );
        const auto goodFaces = mesh.topology.getValidFaces() - badFaces;

        // for open input mesh, let us find only necessary portion on the shell
        auto innerFaces = findInnerShellFacesWithSplits( MeshPart{ mesh, &goodFaces }, resMesh,
            {
                .side = offset > 0 ? Side::Positive : Side::Negative
            } );
        resMesh.topology.deleteFaces( resMesh.topology.getValidFaces() - innerFaces );
        resMesh.pack();
    }

    if ( offset >= 0 )
    {
        // add original mesh to the result with flipping
        resMesh.addPartByMask( mesh, mesh.topology.getValidFaces(), true ); // true = with flipping
    }
    else
    {
        if ( !unsignedOffset ) // in case of unsigned offset (bidirectional shell), resMesh already has opposite normals
            resMesh.topology.flipOrientation();
        // add original mesh to the result without flipping
        resMesh.addPart( mesh );
    }

    resMesh.invalidateCaches();
    return res;
}

Expected<Mesh> offsetPolyline( const Polyline3& polyline, float offset, const OffsetParameters& params /*= {} */ )
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

    // Type::Shell is more efficient in this case
    OffsetParameters p = params;
    p.signDetectionMode = SignDetectionMode::Unsigned;

    return offsetMesh( mesh, offset, p );
}

}
