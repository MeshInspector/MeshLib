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
    bool signPostprocess = !findRegionBoundary( mp.mesh.topology, mp.region ).empty() && !useShell;

    if ( useShell )
        offset = std::abs( offset );

    auto offsetInVoxels = offset / voxelSize;

    auto voxelSizeVector = Vector3f::diagonal( voxelSize );
    // Make grid
    auto grid = ( !useShell ) ?
        // Make level set grid if it is closed
        meshToLevelSet( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 2,
            subprogress( params.callBack, 0.0f, signPostprocess ? 0.33f : 0.5f ) ) :
        // Make distance field grid if it is not closed
        meshToDistanceField( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 2,
                        subprogress( params.callBack, 0.0f, signPostprocess ? 0.33f : 0.5f ) );

    if ( !grid )
        return tl::make_unexpected( "Operation was canceled." );

    if ( signPostprocess )
    {
        auto sp = subprogress( params.callBack, 0.33f, 0.66f );
        std::atomic<bool> keepGoing{ true };
        auto mainThreadId = std::this_thread::get_id();

        FastWindingNumber fwn( mp.mesh );

        auto activeBox = grid->evalActiveVoxelBoundingBox();
        // make dense topology tree to copy its nodes topology to original grid
        std::unique_ptr<openvdb::TopologyTree> topologyTree = std::make_unique<openvdb::TopologyTree>();
        // make it dense
        topologyTree->denseFill( activeBox, {} );
        grid->tree().topologyUnion( *topologyTree ); // after this all voxels should be active and trivial parallelism is ok
        // free topology tree
        topologyTree.reset();

        auto minCoord = activeBox.min();
        auto dims = activeBox.dim();
        VolumeIndexer indexer( Vector3i( dims.x(), dims.y(), dims.z() ) );
        tbb::parallel_for( tbb::blocked_range<size_t>( size_t( 0 ), size_t( activeBox.volume() ) ),
            [&] ( const tbb::blocked_range<size_t>& range )
        {
            auto accessor = grid->getAccessor();
            for ( auto i = range.begin(); i < range.end(); ++i )
            {
                if ( sp && !keepGoing.load( std::memory_order_relaxed ) )
                    break;

                auto pos = indexer.toPos( VoxelId( i ) );
                auto coord = minCoord;
                for ( int j = 0; j < 3; ++j )
                    coord[j] += pos[j];

                auto coord3i = Vector3i( coord.x(), coord.y(), coord.z() );
                auto pointInSpace = voxelSize * Vector3f( coord3i );
                auto windVal = fwn.calc( pointInSpace, 2.0f );
                windVal = std::clamp( 1.0f - 2.0f * windVal, -1.0f, 1.0f );
                if ( windVal < 0.0f )
                    windVal *= -windVal;
                else
                    windVal *= windVal;
                accessor.modifyValue( coord, [windVal] ( float& val ) { val *= windVal; } );
                if ( sp && mainThreadId == std::this_thread::get_id() && !sp( float( i ) / float( range.size() ) ) )
                    keepGoing.store( false, std::memory_order_relaxed );
            }
        }, tbb::static_partitioner() );
        if ( !keepGoing )
            return tl::make_unexpected( "Operation was canceled." );
        grid->pruneGrid( 0.0f );
    }

    // Make offset mesh
    auto newMesh = gridToMesh( std::move( grid ), voxelSizeVector, offsetInVoxels, params.adaptivity, 
        subprogress( params.callBack, signPostprocess ? 0.66f : 0.5f, 1.0f ) );

    if ( !newMesh.has_value() )
        return tl::make_unexpected( "Operation was canceled." );

    // For not closed meshes orientation is flipped on back conversion
    if ( useShell )
        newMesh->topology.flipOrientation();

    return newMesh;
}

tl::expected<Mesh, std::string> doubleOffsetMesh( const MeshPart& mp, float offsetA, float offsetB, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER
    if ( !findRegionBoundary( mp.mesh.topology, mp.region ).empty() )
    {
        spdlog::error( "Only closed meshes allowed for double offset." );
        return tl::make_unexpected( "Only closed meshes allowed for double offset." );
    }
    if ( params.type == OffsetParameters::Type::Shell )
    {
        spdlog::warn( "Cannot use shell for double offset, using offset mode instead." );
    }
    return levelSetDoubleConvertion( mp, AffineXf3f(), params.voxelSize, offsetA, offsetB, params.adaptivity, params.callBack );
}

tl::expected<Mesh, std::string> mcOffsetMesh( const Mesh& mesh, float offset, 
    const BaseOffsetParameters& params, Vector<VoxelId, FaceId> * outMap, bool useSimpleVolume )
{
    MR_TIMER;
    auto meshToLSCb = subprogress( params.callBack, 0.0f, 0.4f );
    if ( !useSimpleVolume )
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
        msParams.signMode = MeshToSimpleVolumeParams::SignDetectionMode::ProjectionNormal;
        msParams.maxDistSq = sqr( absOffset + 2.0f * params.voxelSize );
        msParams.minDistSq = sqr( std::max( absOffset - 2.0f * params.voxelSize, 0.0f ) );
        
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
