#include "MRFastWindingNumber.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRParallelFor.h"
#include "MRBitSetParallelFor.h"
#include "MRVolumeIndexer.h"
#include "MRMeshProject.h"
#include "MRAABBTree.h"

namespace MR
{

FastWindingNumber::FastWindingNumber( const Mesh & mesh ) :
    mesh_( mesh ), 
    tree_( mesh.getAABBTree() )
{
    calcDipoles( dipoles_, tree_, mesh_ );
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

VoidOrErrStr FastWindingNumber::calcFromGrid( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta, ProgressCallback cb )
{
    MR_TIMER

    const size_t size = dims.x * dims.y * dims.z;
    res.resize( size );
    VolumeIndexer indexer( dims );
    if ( !ParallelFor( size_t( 0 ), size, [&]( size_t i )
    {
        auto pos = indexer.toPos( VoxelId( i ) );
        auto coord = minCoord;
        for ( int j = 0; j < 3; ++j )
            coord[j] += pos[j];

        auto coord3i = Vector3i( int( coord.x ), int( coord.y ), int( coord.z ) );
        auto pointInSpace = mult( voxelSize, Vector3f( coord3i ) );
        res[i] = calc_( gridToMeshXf( pointInSpace ), beta );
    }, cb ) )
        return unexpectedOperationCanceled();
    return {};
}

float FastWindingNumber::calcWithDistances( const Vector3f& p, float beta, float maxDistSq, float minDistSq )
{
    const auto sign = calc_( p, beta ) > 0.5f ? -1.f : +1.f;
    return sign * std::sqrt( findProjection( p, mesh_, maxDistSq, nullptr, minDistSq ).distSq );
}

VoidOrErrStr FastWindingNumber::calcFromGridWithDistances( std::vector<float>& res, const Vector3i& dims, const Vector3f& minCoord, const Vector3f& voxelSize, const AffineXf3f& gridToMeshXf, float beta, float maxDistSq, float minDistSq, ProgressCallback cb )
{
    MR_TIMER

    const size_t size = dims.x * dims.y * dims.z;
    res.resize( size );
    VolumeIndexer indexer( dims );

    MeshPart mp( mesh_ );

    if ( !ParallelFor( size_t( 0 ), size, [&]( size_t i )
        {
            auto pos = indexer.toPos( VoxelId( i ) );
            auto coord = minCoord;
            for ( int j = 0; j < 3; ++j )
                coord[j] += pos[j];

            //auto coord3i = Vector3i( int( coord.x ), int( coord.y ), int( coord.z ) );
            const auto pointInSpace = mult( voxelSize, coord );
            const auto transformedPoint = gridToMeshXf( pointInSpace );
            res[i] = calcWithDistances( transformedPoint, beta, maxDistSq, minDistSq );
        }, cb ) )
        return unexpectedOperationCanceled();
    return {};
}

size_t FastWindingNumber::fromVectorHeapBytes( size_t ) const
{
    return 0;
}

size_t FastWindingNumber::selfIntersectionsHeapBytes( const Mesh& ) const
{
    return 0;
}

size_t FastWindingNumber::fromGridHeapBytes( const Vector3i& ) const
{
    return 0;
}

} // namespace MR
