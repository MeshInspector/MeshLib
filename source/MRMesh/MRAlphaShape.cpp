#include "MRAlphaShape.h"
#include "MRPointCloud.h"
#include "MRPointsInBall.h"
#include "MRTriMath.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"

namespace MR
{

void findAlphaShapeNeiTriangles( const PointCloud & cloud, VertId v, float radius,
    Triangulation & appendTris, std::vector<VertId> & tmp, bool onlyLargerVids )
{
    MR_TIMER
    assert( radius > 0 );
    const auto rr = sqr( radius );
    tmp.clear();
    findPointsInBall( cloud, cloud.points[v], 2 * radius,
        [&tmp, v]( VertId n, const Vector3f& )
        {
            if ( v != n )
                tmp.push_back( n );
        } );
    for ( int i = 0; i + 1 < tmp.size(); ++i )
    {
        const auto ni = tmp[i];
        if ( onlyLargerVids && ni < v )
            continue;
        for ( int j = i + 1; j < tmp.size(); ++j )
        {
            const auto nj = tmp[j];
            if ( onlyLargerVids && nj < v )
                continue;
            Vector3f centerPos, centerNeg;
            if ( !circumballCenters( cloud.points[v], cloud.points[ni], cloud.points[nj], radius, centerPos, centerNeg ) )
                continue;
            auto ballEmpty = [&]( const Vector3f & center )
            {
                for ( auto n : tmp )
                {
                    if ( n == ni || n == nj )
                        continue;
                    if ( ( cloud.points[n] - center ).lengthSq() < rr )
                        return false;
                }
                return true;
            };
            if ( ballEmpty( centerPos ) )
                appendTris.push_back( { v, ni, nj } );
            if ( ballEmpty( centerNeg ) )
                appendTris.push_back( { v, nj, ni } );
        }
    }
}

Triangulation findAlphaShapeAllTriangles( const PointCloud & cloud, float radius )
{
    MR_TIMER
    struct ThreadData
    {
        Triangulation tris;
        std::vector<VertId> tmp;
    };

    tbb::enumerable_thread_specific<ThreadData> threadData;
    cloud.getAABBTree(); // to avoid multiple calls to tree construction from parallel region,
                         // which can result that two different vertices will start being processed by one thread

    BitSetParallelFor( cloud.validPoints, [&]( VertId v )
    {
        auto & tls = threadData.local();
        findAlphaShapeNeiTriangles( cloud, v, radius, tls.tris, tls.tmp, true );
    } );

    size_t numTris = 0;
    for ( const auto & tls : threadData )
        numTris += tls.tris.size();

    Triangulation res;
    res.reserve( numTris );
    for ( const auto & tls : threadData )
        res.vec_.insert( end( res ), begin( tls.tris ), end( tls.tris ) );

    /// to avoid dependency on work distribution among threads
    tbb::parallel_sort( begin( res ), end( res ) );

    return res;
}

TEST( MRMesh, AlphaShape )
{
    PointCloud cloud;
    cloud.points.push_back( { 0.5f, 0.5f, 0.1f } ); //0_v
    cloud.points.push_back( { 0.5f, 0.5f, -.1f } ); //1_v
    cloud.points.push_back( { 0,    0,    0 } );    //2_v
    cloud.points.push_back( { 1,    0,    0 } );    //3_v
    cloud.points.push_back( { 0,    1,    0 } );    //4_v
    cloud.validPoints.autoResizeSet( 2_v, 3, true );

    Triangulation tris;
    std::vector<VertId> tmp;

    findAlphaShapeNeiTriangles( cloud, 3_v, 3, tris, tmp, true );
    EXPECT_EQ( tris.size(), 0 );
    findAlphaShapeNeiTriangles( cloud, 4_v, 3, tris, tmp, true );
    EXPECT_EQ( tris.size(), 0 );
    findAlphaShapeNeiTriangles( cloud, 2_v, 3, tris, tmp, true );
    EXPECT_EQ( tris.size(), 2 );

    cloud.validPoints.set( 1_v );
    cloud.invalidateCaches();
    tris.clear();
    findAlphaShapeNeiTriangles( cloud, 2_v, 3, tris, tmp, true );
    EXPECT_EQ( tris.size(), 1 );

    cloud.validPoints.set( 0_v );
    cloud.invalidateCaches();
    tris.clear();
    findAlphaShapeNeiTriangles( cloud, 2_v, 3, tris, tmp, true );
    EXPECT_EQ( tris.size(), 0 );

    const auto allTris = findAlphaShapeAllTriangles( cloud, 3 );
    EXPECT_EQ( allTris.size(), 6 );
}

} //namespace MR
