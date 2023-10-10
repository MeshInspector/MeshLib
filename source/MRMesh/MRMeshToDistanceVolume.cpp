#include "MRMeshToDistanceVolume.h"
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRIsNaN.h"
#include "MRMesh.h"
#include "MRSimpleVolume.h"
#include "MRTimer.h"
#include "MRVolumeIndexer.h"
#include "MRIntersectionPrecomputes.h"
#include "MRFastWindingNumber.h"
#include "MRLine3.h"
#include "MRMeshIntersect.h"
#include "MRParallelFor.h"
#include <tuple>

namespace MR
{

Expected<SimpleVolume, std::string> meshToDistanceVolume( const Mesh& mesh, const MeshToDistanceVolumeParams& params /*= {} */ )
{
    MR_TIMER
    SimpleVolume res;
    res.voxelSize = params.voxelSize;
    res.dims = params.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    // used in Winding rule mode
    const IntersectionPrecomputes<double> precomputedInter( Vector3d::plusX() );
    
    if ( params.signMode == SignDetectionMode::HoleWindingRule )
    {
        auto fwn = params.fwn;
        if ( !fwn )
            fwn = std::make_shared<FastWindingNumber>( mesh );

        auto basis = AffineXf3f::linear( Matrix3f::scale( params.voxelSize ) );
        basis.b = params.origin;
        constexpr float beta = 2;
        if ( auto d = fwn->calcFromGridWithDistances( res.data, res.dims, Vector3f::diagonal( 0.5f ), Vector3f::diagonal( 1.0f ), basis, beta,
            params.maxDistSq, params.minDistSq, params.cb ); !d )
        {
            return unexpected( std::move( d.error() ) );
        }
    }
    else
    {
        if ( !ParallelFor( size_t( 0 ), indexer.size(), [&]( size_t i )
        {
            auto coord = Vector3f( indexer.toPos( VoxelId( i ) ) ) + Vector3f::diagonal( 0.5f );
            auto voxelCenter = params.origin + mult( params.voxelSize, coord );
            float dist{ 0.0f };
            if ( params.signMode != SignDetectionMode::ProjectionNormal )
                dist = std::sqrt( findProjection( voxelCenter, mesh, params.maxDistSq, nullptr, params.minDistSq ).distSq );
            else
            {
                auto s = findSignedDistance( voxelCenter, mesh, params.maxDistSq, params.minDistSq );
                dist = s ? s->dist : cQuietNan;
            }

            if ( !isNanFast( dist ) )
            {
                bool changeSign = false;
                if ( params.signMode == SignDetectionMode::WindingRule )
                {
                    int numInters = 0;
                    rayMeshIntersectAll( mesh, Line3d( Vector3d( voxelCenter ), Vector3d::plusX() ),
                        [&numInters] ( const MeshIntersectionResult& ) mutable
                    {
                        ++numInters;
                        return true;
                    } );
                    changeSign = numInters % 2 == 1; // inside
                }
                if ( changeSign )
                    dist = -dist;
            }
            res.data[i] = dist;
        }, params.cb ) )
            return unexpectedOperationCanceled();
    }

    std::tie( res.min, res.max ) = parallelMinMax( res.data );
    return res;
}

Expected<SimpleVolume, std::string> meshRegionToIndicatorVolume( const Mesh& mesh, const FaceBitSet& region,
    float offset, const DistanceVolumeParams& params )
{
    MR_TIMER
    if ( !region.any() )
    {
        assert( false );
        return unexpected( "empty region" );
    }

    SimpleVolume res;
    res.voxelSize = params.voxelSize;
    res.dims = params.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    AABBTree regionTree( { mesh, &region } );
    const FaceBitSet notRegion = mesh.topology.getValidFaces() - region;
    //TODO: check that notRegion is not empty
    AABBTree notRegionTree( { mesh, &notRegion } );
    
    const auto voxelSize = std::max( { params.voxelSize.x, params.voxelSize.y, params.voxelSize.z } );

    if ( !ParallelFor( size_t( 0 ), indexer.size(), [&]( size_t i )
    {
        const auto coord = Vector3f( indexer.toPos( VoxelId( i ) ) ) + Vector3f::diagonal( 0.5f );
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

} //namespace MR
#endif
