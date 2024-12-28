#include "MRFastWindingNumber.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRVolumeIndexer.h"
#include "MRMeshProject.h"
#include "MRAABBTree.h"
#include "MRDipole.h"
#include "MRIsNaN.h"
#include "MRDistanceToMeshOptions.h"

namespace MR
{

FastWindingNumber::FastWindingNumber( const Mesh & mesh ) :
    mesh_( mesh ),
    tree_( mesh.getAABBTree() ),
    dipoles_( mesh.getDipoles() )
{
}

inline float FastWindingNumber::calc_( const Vector3f & q, float beta, FaceId skipFace ) const
{
    return calcFastWindingNumber( dipoles_, tree_, mesh_, q, beta, skipFace );
}

void FastWindingNumber::calcFromVector( std::vector<float>& res, const std::vector<Vector3f>& points, float beta, FaceId skipFace )
{
    res.resize( points.size() );
    ParallelFor( points, [&]( size_t i )
    {
        res[i] = calc_( points[i], beta, skipFace );
    } );
}

bool FastWindingNumber::calcSelfIntersections( FaceBitSet& res, float beta, ProgressCallback cb )
{
    res.resize( mesh_.topology.faceSize() );
    return BitSetParallelFor( mesh_.topology.getValidFaces(), [&] ( FaceId f )
    {
        auto wn = calc_( mesh_.triCenter( f ), beta, f );
        if ( wn < 0 || wn > 1 )
            res.set( f );
    }, cb );
}

Expected<void> FastWindingNumber::calcFromGrid( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, float beta, ProgressCallback cb )
{
    MR_TIMER

    VolumeIndexer indexer( dims );
    res.resize( indexer.size() );

    if ( !ParallelFor( 0_vox, indexer.endId(), [&]( VoxelId i )
    {
        res[i] = calc_( gridToMeshXf( Vector3f( indexer.toPos( i ) ) ), beta );
    }, cb ) )
        return unexpectedOperationCanceled();
    return {};
}

float FastWindingNumber::calcWithDistances( const Vector3f& p, const DistanceToMeshOptions& options )
{
    auto resSq = findProjection( p, mesh_, options.maxDistSq, nullptr, options.minDistSq ).distSq;
    if ( options.nullOutsideMinMax && ( resSq < options.minDistSq || resSq >= options.maxDistSq ) ) // note that resSq == minDistSq (e.g. == 0) is a valid situation
        return cQuietNan;
    const auto sign = calc_( p, options.windingNumberBeta ) > options.windingNumberThreshold ? -1.f : +1.f;
    return sign * std::sqrt( resSq );
}

Expected<void> FastWindingNumber::calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const AffineXf3f& gridToMeshXf, const DistanceToMeshOptions& options, const ProgressCallback& cb )
{
    MR_TIMER

    VolumeIndexer indexer( dims );
    res.resize( indexer.size() );

    MeshPart mp( mesh_ );

    if ( !ParallelFor( 0_vox, indexer.endId(), [&]( VoxelId i )
        {
            const auto transformedPoint = gridToMeshXf( Vector3f( indexer.toPos( i ) ) );
            res[i] = calcWithDistances( transformedPoint, options );
        }, cb ) )
        return unexpectedOperationCanceled();
    return {};
}

} // namespace MR
