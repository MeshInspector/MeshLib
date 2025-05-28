#include "MRMeshToDistanceVolume.h"
#include "MRVDBConversions.h"
#include "MRMesh/MRIsNaN.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRParallelMinMax.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRPointsToMeshProjector.h"
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
