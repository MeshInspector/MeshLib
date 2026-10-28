#include "MRBestPlaneVolume.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRSymMatrix3.h"
#include "MRMatrix3.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <array>

namespace MR
{

namespace
{

struct FaceData
{
    Vector3f center;
    Vector3f dirArea;
    float area = 0;
    std::array<FaceId, 3> neis; // invalid on boundary
};

struct ThreadData
{
    FaceBitSet visited;
    std::vector<FaceId> touched;
    std::vector<FaceId> stack;
};

} //anonymous namespace

Expected<VertScalars> computeBestPlaneVolumes( const Mesh& mesh, float radius, const ProgressCallback& cb )
{
    MR_TIMER;
    const auto& topology = mesh.topology;

    Vector<FaceData, FaceId> faces( topology.faceSize() );
    if ( !BitSetParallelFor( topology.getValidFaces(), [&]( FaceId f )
    {
        auto& fd = faces[f];
        int i = 0;
        Vector3f p[3];
        for ( auto e : leftRing( topology, f ) )
        {
            assert( i < 3 );
            p[i] = mesh.points[topology.org( e )];
            fd.neis[i++] = topology.right( e );
        }
        fd.center = ( p[0] + p[1] + p[2] ) / 3.0f;
        fd.dirArea = 0.5f * cross( p[1] - p[0], p[2] - p[0] );
        fd.area = fd.dirArea.length();
    }, subprogress( cb, 0.0f, 0.1f ) ) )
        return unexpectedOperationCanceled();

    const float rSq = sqr( radius );
    const double rCube = double( radius ) * radius * radius;
    VertScalars res( topology.vertSize(), 0.0f );
    tbb::enumerable_thread_specific<ThreadData> tls;
    if ( !BitSetParallelFor( topology.getValidVerts(), tls, [&]( VertId v, ThreadData& td )
    {
        if ( td.visited.size() != faces.size() )
            td.visited.resize( faces.size() );
        const auto p0 = mesh.points[v];
        auto tryAdd = [&]( FaceId f )
        {
            if ( !f || td.visited.test_set( f ) )
                return;
            td.touched.push_back( f );
            if ( ( faces[f].center - p0 ).lengthSq() <= rSq )
                td.stack.push_back( f );
        };
        for ( auto e : orgRing( topology, v ) )
            tryAdd( topology.left( e ) );

        // all coordinates are relative to p0;
        // plane (n, d): sum_f ( dot( n, c_f ) - d ) * dot( n, D_f ) = dot( n, sum_f( c_f * D_f^T ) * n ) - d * dot( n, sum_f D_f )
        double sumArea = 0;
        Vector3d sumCenter; // area-weighted
        SymMatrix3d sumCenterSq; // area-weighted
        Matrix3d sumCenterDirArea = Matrix3d::zero();
        Vector3d sumDirArea;
        while ( !td.stack.empty() )
        {
            const auto& fd = faces[td.stack.back()];
            td.stack.pop_back();
            const Vector3d c( fd.center - p0 );
            const Vector3d d( fd.dirArea );
            const double a = fd.area;
            sumArea += a;
            sumCenter += a * c;
            sumCenterSq += outerSquare( a, c );
            sumCenterDirArea += outer( c, d );
            sumDirArea += d;
            for ( auto nf : fd.neis )
                tryAdd( nf );
        }
        for ( auto f : td.touched )
            td.visited.reset( f );
        td.touched.clear();

        if ( sumArea <= 0 )
            return;
        const auto centroid = sumCenter / sumArea;
        Matrix3d eigenvectors;
        ( sumCenterSq - outerSquare( 1 / sumArea, sumCenter ) ).eigens( &eigenvectors );
        const auto n = eigenvectors.x;
        res[v] = float( ( dot( n, sumCenterDirArea * n ) - dot( n, centroid ) * dot( n, sumDirArea ) ) / rCube );
    }, subprogress( cb, 0.1f, 1.0f ) ) )
        return unexpectedOperationCanceled();

    return res;
}

} //namespace MR
