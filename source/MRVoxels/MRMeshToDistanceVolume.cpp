#include "MRMeshToDistanceVolume.h"
#include "MRVDBConversions.h"
#include "MRVDBFloatGrid.h"
#include "MRMesh/MRIsNaN.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRParallelMinMax.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRPointsToMeshProjector.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRLine.h"
#include "MRPch/MROpenVDB.h"
#include "MRPch/MRTBB.h"
#include <algorithm>
#include <tuple>

namespace MR
{

Expected<SimpleVolumeMinMax> meshToDistanceVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& cParams /*= {} */ )
{
    MR_TIMER;
    if ( cParams.dist.signMode == SignDetectionMode::OpenVDB )
    {
        MeshToVolumeParams m2vPrams
        {
            .type = MeshToVolumeParams::Type::Signed,
            .voxelSize = cParams.vol.voxelSize,
            // SimpleVolume and VdbVolume are shifted on half voxel relative one another, see also VoxelsVolumeAccessor::shift()
            .worldXf = AffineXf3f::translation( -cParams.vol.origin - 0.5f * cParams.vol.voxelSize ),
            .cb = subprogress( cParams.vol.cb, 0.0f, 0.8f )
        };
        assert( cParams.dist.maxDistSq < FLT_MAX ); // the amount of work is proportional to maximal distance
        if ( cParams.dist.maxDistSq < FLT_MAX )
        {
            m2vPrams.surfaceOffset = std::sqrt( cParams.dist.maxDistSq )
                / std::min( { cParams.vol.voxelSize.x, cParams.vol.voxelSize.y, cParams.vol.voxelSize.z } );
        }
        return meshToDistanceVdbVolume( mp, m2vPrams ).and_then(
            [&cParams]( VdbVolume && vdbVolume )
            {
                return vdbVolumeToSimpleVolume( vdbVolume, Box3i{ Vector3i( 0, 0, 0 ), cParams.vol.dimensions }, subprogress( cParams.vol.cb, 0.8f, 1.0f ) );
            } );
    }

    auto params = cParams;
    if ( params.dist.signMode == SignDetectionMode::HoleWindingRule )
    {
        SimpleVolumeMinMax res;
        res.voxelSize = params.vol.voxelSize;
        res.dims = params.vol.dimensions;
        VolumeIndexer indexer( res.dims );
        res.data.resize( indexer.size() );

        if ( !params.fwn )
            params.fwn = std::make_shared<FastWindingNumber>( mp.mesh );
        assert( !mp.region ); // only whole mesh is supported for now
        auto basis = AffineXf3f( Matrix3f::scale( params.vol.voxelSize ), params.vol.origin + 0.5f * params.vol.voxelSize );
        if ( auto d = params.fwn->calcFromGridWithDistances( res.data.vec_, res.dims, basis, params.dist, params.vol.cb ); !d )
        {
            return unexpected( std::move( d.error() ) );
        }
        std::tie( res.min, res.max ) = parallelMinMax( res.data );
        return res;
    }

    const auto func = meshToDistanceFunctionVolume( mp, params );
    return functionVolumeToSimpleVolume( func, params.vol.cb );

}

FunctionVolume meshToDistanceFunctionVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& params )
{
    MR_TIMER;
    assert( params.dist.signMode != SignDetectionMode::OpenVDB );

    // prepare all trees before returned function will be called from parallel threads
    mp.mesh.getAABBTree();
    if ( params.dist.signMode == SignDetectionMode::HoleWindingRule )
        mp.mesh.getDipoles();

    return FunctionVolume
    {
        .data = [params, mp] ( const Vector3i& pos ) -> float
        {
            const auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
            const auto voxelCenter = params.vol.origin + mult( params.vol.voxelSize, coord );
            auto dist = signedDistanceToMesh( mp, voxelCenter, params.dist );
            return dist ? *dist : cQuietNan;
        },
        .dims = params.vol.dimensions,
        .voxelSize = params.vol.voxelSize
    };
}

Expected<SimpleBinaryVolume> makeCloseToMeshVolume( const MeshPart& mp, const CloseToMeshVolumeParams& params )
{
    MR_TIMER;
    assert( params.closeDist >= 0 );
    SimpleBinaryVolume res;
    res.voxelSize = params.vol.voxelSize;
    res.dims = params.vol.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size(), false );

    mp.mesh.getAABBTree();
    if ( !BitSetParallelForAll( res.data, [&, closeDistSq = sqr( params.closeDist )] ( VoxelId i )
    {
        const auto pos = indexer.toPos( i );
        const auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
        const auto voxelCenter = params.vol.origin + mult( params.vol.voxelSize, coord );
        const auto anythingWithinCloseDist = findProjection( voxelCenter, mp, closeDistSq, params.meshToWorld, closeDistSq );
        if ( anythingWithinCloseDist )
            res.data.set( i );
    }, params.vol.cb ) )
        return unexpectedOperationCanceled();

    return res;
}

Expected<SimpleBinaryVolume> makeInsideMeshVolume( const MeshPart& mp, const DistanceVolumeParams& params, InsideMeshRule rule )
{
    MR_TIMER;
    if ( !mp.mesh.topology.isClosed( mp.region ) )
        return unexpected( "Only closed mesh can be converted to inside volume" );

    SimpleBinaryVolume res;
    res.voxelSize = params.voxelSize;
    res.dims = params.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size(), false );
    if ( res.dims.x <= 0 || res.dims.y <= 0 || res.dims.z <= 0 )
        return res;

    // one ray along X through the voxel centers of each row; precise predicates in rayMeshIntersectAll( Line3d )
    // guarantee that every ray crosses the closed surface an even number of times with zero total winding;
    // each task processes 64 whole rows, which occupy whole blocks of the bit set, so no two tasks write in the same block
    mp.mesh.getAABBTree();
    struct Hit
    {
        float t;
        int winding; // +1 if the ray enters the mesh here, -1 if it leaves
    };
    const auto isInside = [rule] ( int numCrossings, int winding )
    {
        return rule == InsideMeshRule::OddCrossings ? numCrossings % 2 == 1 : winding > 0;
    };
    tbb::enumerable_thread_specific<std::vector<Hit>> hitsPerThread;
    const size_t numRows = size_t( res.dims.y ) * res.dims.z;
    if ( !ParallelFor( size_t( 0 ), ( numRows + 63 ) / 64, hitsPerThread, [&] ( size_t chunk, std::vector<Hit> & hits )
    {
        for ( size_t row = chunk * 64; row < std::min( numRows, chunk * 64 + 64 ); ++row )
        {
            const auto y = int( row % res.dims.y );
            const auto z = int( row / res.dims.y );
            const Vector3d start( params.origin.x,
                params.origin.y + ( y + 0.5 ) * params.voxelSize.y,
                params.origin.z + ( z + 0.5 ) * params.voxelSize.z );
            hits.clear();
            rayMeshIntersectAll( mp, Line3d( start, Vector3d( 1, 0, 0 ) ), [&hits] ( const MeshIntersectionResult & isec, bool fromFront )
            {
                hits.push_back( { isec.distanceAlongLine, fromFront ? 1 : -1 } );
                return true;
            }, -DBL_MAX, DBL_MAX );
            std::sort( hits.begin(), hits.end(), [] ( const Hit & a, const Hit & b ) { return a.t < b.t; } );

            // walking the ray, set the voxels with centers between the hits where the rule switches to inside and back
            const auto firstVoxel = [&] ( float t ) { return std::clamp( (int)std::ceil( t / params.voxelSize.x - 0.5 ), 0, res.dims.x ); };
            int numCrossings = 0, winding = 0, xBeg = 0;
            for ( const auto & hit : hits )
            {
                const bool wasInside = isInside( numCrossings, winding );
                ++numCrossings;
                winding += hit.winding;
                const bool nowInside = isInside( numCrossings, winding );
                if ( !wasInside && nowInside )
                    xBeg = firstVoxel( hit.t );
                else if ( wasInside && !nowInside )
                {
                    const auto xEnd = firstVoxel( hit.t );
                    if ( xBeg < xEnd )
                        res.data.set( VoxelId( row * res.dims.x + xBeg ), xEnd - xBeg, true );
                }
            }
        }
    }, params.cb ) )
        return unexpectedOperationCanceled();

    return res;
}

Expected<SimpleBinaryVolume> makeInsideMeshVolumeVdb( const MeshPart& mp, const DistanceVolumeParams& params )
{
    MR_TIMER;
    if ( !mp.mesh.topology.isClosed( mp.region ) )
        return unexpected( "Only closed mesh can be converted to inside volume" );

    // SimpleVolume and VdbVolume are shifted on half voxel relative one another, see also VoxelsVolumeAccessor::shift()
    const auto grid = meshToLevelSet( mp, AffineXf3f::translation( -params.origin - 0.5f * params.voxelSize ),
        params.voxelSize, 0.5f, subprogress( params.cb, 0.0f, 0.8f ) );
    if ( !grid )
        return unexpectedOperationCanceled();

    SimpleBinaryVolume res;
    res.voxelSize = params.voxelSize;
    res.dims = params.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size(), false );

    tbb::enumerable_thread_specific accessorPerThread( grid->getConstAccessor() );
    if ( !BitSetParallelForAll( res.data, [&] ( VoxelId i )
    {
        const auto pos = indexer.toPos( i );
        if ( accessorPerThread.local().getValue( openvdb::Coord( pos.x, pos.y, pos.z ) ) < 0 )
            res.data.set( i );
    }, subprogress( params.cb, 0.8f, 1.0f ) ) )
        return unexpectedOperationCanceled();

    return res;
}

Expected<SimpleVolumeMinMax> meshRegionToIndicatorVolume( const Mesh& mesh, const FaceBitSet& region,
    float offset, const DistanceVolumeParams& params )
{
    MR_TIMER;
    if ( !region.any() )
    {
        assert( false );
        return unexpected( "empty region" );
    }

    SimpleVolumeMinMax res;
    res.voxelSize = params.voxelSize;
    res.dims = params.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    AABBTree regionTree( { mesh, &region } );
    const FaceBitSet notRegion = mesh.topology.getValidFaces() - region;
    //TODO: check that notRegion is not empty
    AABBTree notRegionTree( { mesh, &notRegion } );

    const auto voxelSize = std::max( { params.voxelSize.x, params.voxelSize.y, params.voxelSize.z } );

    if ( !ParallelFor( 0_vox, indexer.endId(), [&]( VoxelId i )
    {
        const auto coord = Vector3f( indexer.toPos( i ) ) + Vector3f::diagonal( 0.5f );
        auto voxelCenter = params.origin + mult( params.voxelSize, coord );

        // minimum of given offset distance parameter and the distance to not-region part of mesh
        const auto distToNotRegion = std::sqrt( findProjectionSubtree( voxelCenter, mesh, notRegionTree, sqr( offset ) ).distSq );

        const auto maxDistSq = sqr( distToNotRegion + voxelSize );
        const auto minDistSq = sqr( std::max( distToNotRegion - voxelSize, 0.0f ) );
        const auto distToRegion = std::sqrt( findProjectionSubtree( voxelCenter, mesh, regionTree, maxDistSq, nullptr, minDistSq ).distSq );

        res.data[i] = distToRegion - distToNotRegion;
    }, params.cb ) )
        return unexpectedOperationCanceled();

    std::tie( res.min, res.max ) = parallelMinMax( res.data );

    return res;
}

Expected<std::array<SimpleVolumeMinMax, 3>> meshToDirectionVolume( const MeshToDirectionVolumeParams& params )
{
    MR_TIMER;
    VolumeIndexer indexer( params.vol.dimensions );
    std::vector<MeshProjectionResult> projs;

    auto getPoint = [&indexer, &params] ( VoxelId i )
    {
        const auto c = Vector3f( indexer.toPos( i ) ) + Vector3f::diagonal( 0.5f );
        return params.vol.origin + mult( params.vol.voxelSize, c );
    };

    {
        std::vector<Vector3f> points( indexer.size() );
        for ( auto i = VoxelId( size_t( 0 ) ); i < indexer.size(); ++i )
        {
            points[i] = getPoint( i );
        }
        params.projector->findProjections( projs, points );
    }

    std::array<SimpleVolumeMinMax, 3> res;
    for ( auto& v : res )
    {
        v.voxelSize = params.vol.voxelSize;
        v.dims = params.vol.dimensions;
        v.data.resize( indexer.size() );
    }

    for ( auto i = VoxelId( size_t( 0 ) ); i < indexer.size(); ++i )
    {
        const auto d = ( getPoint( i ) - projs[i].proj.point ).normalized();
        res[0].data[i] = d.x;
        res[1].data[i] = d.y;
        res[2].data[i] = d.z;
    }

    for ( auto& v : res )
    {
        std::tie( v.min, v.max ) = parallelMinMax( v.data );
    }

    return res;
}


} //namespace MR
