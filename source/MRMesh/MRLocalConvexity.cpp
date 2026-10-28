#include "MRLocalConvexity.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRPointCloudRadius.h"
#include "MRPointsInBall.h"
#include "MRBestFit.h"
#include "MRConstants.h"
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

// computed once per triangle, since each triangle is visited from the regions of many vertices
struct FaceData
{
    Vector3f center;
    Vector3f dirArea;
    std::array<VertId, 3> orgs; // origins of the edges having this face on the left
    std::array<FaceId, 3> neis; // right faces of the same edges, invalid on boundary
};

struct ThreadData
{
    FaceBitSet visited;
    std::vector<FaceId> touched;
    std::vector<FaceId> stack;
};

} //anonymous namespace

Expected<VertScalars> computeLocalConvexity( const Mesh& mesh, float radius, const ProgressCallback& cb )
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
            fd.orgs[i] = topology.org( e );
            p[i] = mesh.points[fd.orgs[i]];
            fd.neis[i++] = topology.right( e );
        }
        fd.center = ( p[0] + p[1] + p[2] ) / 3.0f;
        fd.dirArea = 0.5f * cross( p[1] - p[0], p[2] - p[0] );
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
        auto inRegion = [&]( FaceId f )
        {
            return f && ( faces[f].center - p0 ).lengthSq() <= rSq;
        };
        auto tryAdd = [&]( FaceId f )
        {
            if ( td.visited.test_set( f ) )
                return;
            td.touched.push_back( f );
            td.stack.push_back( f );
        };
        for ( auto e : orgRing( topology, v ) )
            if ( auto f = topology.left( e ); inRegion( f ) )
                tryAdd( f );

        // all coordinates are relative to p0;
        // plane (n, d): sum_f ( dot( n, c_f ) - d ) * dot( n, D_f ) = dot( n, sum_f( c_f * D_f^T ) * n ) - d * dot( n, sum_f D_f )
        int numBdPoints = 0;
        Vector3d sumBdPoint;
        SymMatrix3d sumBdPointSq;
        Matrix3d sumCenterDirArea = Matrix3d::zero();
        Vector3d sumDirArea;
        while ( !td.stack.empty() )
        {
            const auto& fd = faces[td.stack.back()];
            td.stack.pop_back();
            const Vector3d d( fd.dirArea );
            sumCenterDirArea += outer( Vector3d( fd.center - p0 ), d );
            sumDirArea += d;
            for ( int i = 0; i < 3; ++i )
            {
                if ( inRegion( fd.neis[i] ) )
                {
                    tryAdd( fd.neis[i] );
                    continue;
                }
                // each region boundary edge contributes its origin, so each boundary vertex is taken once per boundary loop
                const Vector3d q( mesh.points[fd.orgs[i]] - p0 );
                ++numBdPoints;
                sumBdPoint += q;
                sumBdPointSq += outerSquare( q );
            }
        }
        for ( auto f : td.touched )
            td.visited.reset( f );
        td.touched.clear();

        if ( numBdPoints <= 0 )
            return;
        const auto centroid = sumBdPoint / double( numBdPoints );
        Matrix3d eigenvectors;
        ( sumBdPointSq - outerSquare( 1.0 / numBdPoints, sumBdPoint ) ).eigens( &eigenvectors );
        const auto n = eigenvectors.x;
        res[v] = float( ( dot( n, sumCenterDirArea * n ) - dot( n, centroid ) * dot( n, sumDirArea ) ) / rCube );
    }, subprogress( cb, 0.1f, 1.0f ) ) )
        return unexpectedOperationCanceled();

    return res;
}

Expected<VertScalars> computeLocalConvexity( const PointCloud& cloud, float radius, const ProgressCallback& cb )
{
    MR_TIMER;
    if ( !cloud.hasNormals() )
        return unexpected( "Point cloud has no normals" );
    // plane points are taken from the shell around the sphere of given radius, as wide as the average distance between neighbor points
    const float shell = findAvgPointsRadius( cloud, 3 );
    cloud.getAABBTree(); // build it before the parallel region

    const float rSq = sqr( radius );
    const float innerSq = sqr( std::max( 0.0f, radius - shell / 2 ) );
    const float outerSq = sqr( radius + shell / 2 );
    VertScalars res( cloud.points.size(), 0.0f );
    tbb::enumerable_thread_specific<std::vector<VertId>> tls;
    if ( !BitSetParallelFor( cloud.validPoints, tls, [&]( VertId v, std::vector<VertId>& ball )
    {
        const auto p0 = cloud.points[v];
        ball.clear();
        PointAccumulator planeAcc; // relative to p0
        findPointsInBall( cloud, { p0, outerSq }, [&]( const PointsProjectionResult & found, const Vector3f&, Ball3f & )
        {
            if ( found.distSq <= rSq )
                ball.push_back( found.vId );
            if ( found.distSq >= innerSq )
                planeAcc.addPoint( cloud.points[found.vId] - p0 );
            return Processing::Continue;
        } );
        if ( !planeAcc.valid() )
            return;
        const auto plane = planeAcc.getBestPlane();
        double sum = 0, sumAbsCos = 0;
        for ( auto u : ball )
        {
            const auto c = dot( plane.n, Vector3d( cloud.normals[u] ) );
            sum += plane.distance( Vector3d( cloud.points[u] - p0 ) ) * c;
            sumAbsCos += std::abs( c );
        }
        // each point's projected area is taken as pi * radius^2 / sumAbsCos
        if ( sumAbsCos > 0 )
            res[v] = float( PI * sum / ( sumAbsCos * radius ) );
    }, cb ) )
        return unexpectedOperationCanceled();

    return res;
}

} //namespace MR
