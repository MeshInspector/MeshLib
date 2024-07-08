#include "MRMeshToDistanceVolume.h"
#include "MRIsNaN.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRVolumeIndexer.h"
#include "MRFastWindingNumber.h"
#include "MRLine3.h"
#include "MRMeshIntersect.h"
#include "MRParallelFor.h"
#include "MRAABBTree.h"
#include <tuple>

namespace
{

using namespace MR;

float signedDistanceToMesh( const MeshPart& mesh, const Vector3f& p, SignDetectionMode signMode, float maxDistSq, float minDistSq )
{
    float dist;
    if ( signMode != SignDetectionMode::ProjectionNormal )
    {
        dist = std::sqrt( findProjection( p, mesh, maxDistSq, nullptr, minDistSq ).distSq );
    }
    else
    {
        const auto s = findSignedDistance( p, mesh, maxDistSq, minDistSq );
        dist = s ? s->dist : cQuietNan;
    }
    if ( isNanFast( dist ) )
        return dist;

    if ( signMode == SignDetectionMode::WindingRule )
    {
        const Line3d ray( Vector3d( p ), Vector3d::plusX() );
        int count = 0;
        rayMeshIntersectAll( mesh, ray, [&count] ( auto&& ) { ++count; return true; } );
        if ( count % 2 == 1 ) // inside
            dist = -dist;
    }

    return dist;
}

template <typename T>
struct MinMax
{
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::lowest();

    void update( T v )
    {
        if ( v < min )
            min = v;
        if ( max < v )
            max = v;
    }

    static MinMax<T> merge( const MinMax<T>& a, const MinMax<T>& b )
    {
        return {
            .min = std::min( a.min, b.min ),
            .max = std::max( a.max, b.max ),
        };
    }
};

} // namespace

namespace MR
{

Expected<SimpleVolume, std::string> meshToDistanceVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& params /*= {} */ )
{
    MR_TIMER
    assert( params.signMode != SignDetectionMode::OpenVDB );
    SimpleVolume res;
    res.voxelSize = params.voxelSize;
    res.dims = params.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    if ( params.signMode == SignDetectionMode::HoleWindingRule )
    {
        assert( !mp.region ); // only whole mesh is supported for now
        auto fwn = params.fwn;
        if ( !fwn )
            fwn = std::make_shared<FastWindingNumber>( mp.mesh );

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
                dist = std::sqrt( findProjection( voxelCenter, mp, params.maxDistSq, nullptr, params.minDistSq ).distSq );
            else
            {
                auto s = findSignedDistance( voxelCenter, mp, params.maxDistSq, params.minDistSq );
                dist = s ? s->dist : cQuietNan;
            }

            if ( !isNanFast( dist ) )
            {
                bool changeSign = false;
                if ( params.signMode == SignDetectionMode::WindingRule )
                {
                    int numInters = 0;
                    rayMeshIntersectAll( mp, Line3d( Vector3d( voxelCenter ), Vector3d::plusX() ),
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

Expected<FunctionVolume> meshToDistanceFunctionVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& params )
{
    MR_TIMER
    assert( params.signMode != SignDetectionMode::OpenVDB );

    FunctionVolume result {
        .dims = params.dimensions,
        .voxelSize = params.voxelSize,
    };
    if ( params.signMode == SignDetectionMode::HoleWindingRule )
    {
        assert( !mp.region ); // only whole mesh is supported for now
        // CUDA-based implementation is useless for FunctionVolume for obvious reasons
        // using default implementation
        auto fwn = std::make_shared<FastWindingNumber>( mp.mesh );
        result.data = [params, fwn] ( const Vector3i& pos ) mutable -> float
        {
            const auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
            const auto voxelCenter = params.origin + mult( params.voxelSize, coord );
            constexpr float beta = 2;
            return fwn->calcWithDistances( voxelCenter, beta, params.maxDistSq, params.minDistSq );
        };
    }
    else
    {
        result.data = [params, mp = MeshPart( mp.mesh )] ( const Vector3i& pos ) -> float
        {
            const auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
            const auto voxelCenter = params.origin + mult( params.voxelSize, coord );
            return signedDistanceToMesh( mp, voxelCenter, params.signMode, params.maxDistSq, params.minDistSq );
        };
    }

    return result;
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
