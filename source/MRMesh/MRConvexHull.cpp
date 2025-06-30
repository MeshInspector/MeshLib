#include "MRConvexHull.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRLine3.h"
#include "MRPlane3.h"
#include "MRMeshBuilder.h"
#include "MREdgeIterator.h"
#include "MRRingIterator.h"
#include "MRphmap.h"
#include "MRMeshFixer.h"
#include "MRHeap.h"
#include "MRTimer.h"
#include "MRTorus.h"
#include "MRGTest.h"

namespace MR
{

static VertId getMinXyzVertex( const VertCoords & points, const VertBitSet & validPoints )
{
    VertId res;
    Vector3f p;
    for ( VertId v : validPoints )
    {
        Vector3f t = points[v];
        if ( !res || std::tie( t.x, t.y, t.z ) < std::tie( p.x, p.y, p.z ) )
        {
            res = v;
            p = t;
        }
    }

    return res;
}

static VertId getFurthestVertexFromPoint( const VertCoords & points, const VertBitSet & validPoints, const Vector3f & p )
{
    VertId res;
    float maxDistSq = 0;
    for ( VertId v : validPoints )
    {
        const auto distSq = distanceSq( points[v], p );
        if ( !res || maxDistSq < distSq )
        {
            res = v;
            maxDistSq = distSq;
        }
    }

    return res;
}

static VertId getFurthestVertexFromLine(  const VertCoords & points, const VertBitSet & validPoints, const Line3f & line )
{
    VertId res;
    float maxDistSq = 0;
    for ( VertId v : validPoints )
    {
        const auto distSq = line.distanceSq( points[v] );
        if ( !res || maxDistSq < distSq )
        {
            res = v;
            maxDistSq = distSq;
        }
    }

    return res;
}

// return false if this must be flipped to restore model convexity
static bool goodConvexEdge( Mesh & mesh, EdgeId edge )
{
    VertId a, b, c, d;
    mesh.topology.getLeftTriVerts( edge, a, c, d );
    assert( a != c );
    b = mesh.topology.dest( mesh.topology.prev( edge ) );
    if( b == d )
        return true; // consider condition satisfied to avoid creation of loop edges

    const Vector3d ap{ mesh.points[a] };
    const Vector3d bp{ mesh.points[b] };
    const Vector3d cp{ mesh.points[c] };
    const Vector3d dp{ mesh.points[d] };
    return dot( cross( cp - ap, dp - ap ), bp - ap ) <= 0; // already convex at testEdge
}

static void makeConvexOriginRing( Mesh & mesh, EdgeId e )
{
    mesh.topology.flipEdgesIn( e, [&]( EdgeId testEdge )
    {
        return !goodConvexEdge( mesh, testEdge );
    } );
}

const double NoDist = -1.0;

Mesh makeConvexHull( const VertCoords & points, const VertBitSet & validPoints )
{
    MR_TIMER;
    Mesh res;
    if ( validPoints.count() < 3 )
        return res;

    const VertId v0 = getMinXyzVertex( points, validPoints );
    const VertId v1 = getFurthestVertexFromPoint( points, validPoints, points[v0] );
    const VertId v2 = getFurthestVertexFromLine( points, validPoints, Line3f{ points[v0], ( points[v1] - points[v0] ).normalized() } );

    Triangulation t =
    {
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 1_v }
    };
    res.topology = MeshBuilder::fromTriangles( t );

    res.points.push_back( points[v0] );
    res.points.push_back( points[v1] );
    res.points.push_back( points[v2] );

    // face of res-mesh to original points above it
    HashMap<FaceId, std::vector<VertId>> face2verts;
    Heap<double, FaceId> queue{ 2, NoDist };

    // separate all remaining points as above face #0 or face #1
    {
        const auto pl0 = res.getPlane3d( 0_f );
        std::vector<VertId> vs0, vs1;
        double maxDist0 = NoDist, maxDist1 = NoDist;
        for ( VertId v : validPoints )
        {
            if ( v == v0 || v == v1 || v == v2 )
                continue;
            const auto dist = pl0.distance( Vector3d{ points[v] } );
            if ( dist >= 0 )
            {
                vs0.push_back( v );
                maxDist0 = std::max( maxDist0, dist );
            }
            else
            {
                vs1.push_back( v );
                maxDist1 = std::max( maxDist1, -dist );
            }
        }
        queue.setValue( 0_f, maxDist0 );
        queue.setValue( 1_f, maxDist1 );
        if ( !vs0.empty() )
            face2verts[0_f] = std::move( vs0 );
        if ( !vs1.empty() )
            face2verts[1_f] = std::move( vs1 );
    }

    struct FacePoints
    {
        FaceId face; // of res-mesh
        Plane3d plane; // with normal outside
        std::vector<VertId> verts; // above that face
        double maxDist = NoDist;
    };
    std::vector<FacePoints> newFp;

    while ( queue.top().val > NoDist )
    {
        const auto myFace = queue.top().id;
        auto it = face2verts.find( myFace );
        if ( it == face2verts.end() )
        {
            queue.setSmallerValue( myFace, NoDist );
            continue;
        }
        auto myverts = std::move( it->second );
        face2verts.erase( it );

        if ( myverts.empty() || !res.topology.hasFace( myFace ) )
        {
            queue.setSmallerValue( myFace, NoDist );
            continue;
        }

        VertId topmostVert;
        double maxDist = 0;
        const auto pl = res.getPlane3d( myFace );
        for ( auto v : myverts )
        {
            auto dist = pl.distance( Vector3d{ points[v] } );
            if ( !topmostVert || dist > maxDist )
            {
                topmostVert = v;
                maxDist = dist;
            }
        }
        if ( !topmostVert )
        {
            queue.setSmallerValue( myFace, NoDist );
            continue;
        }
        auto newv = res.splitFace( myFace, points[topmostVert] );

        makeConvexOriginRing( res, res.topology.edgeWithOrg( newv ) );
        queue.resize( (int)res.topology.faceSize() );

        for ( EdgeId e : orgRing( res.topology, newv ) )
        {
            auto it1 = face2verts.find( res.topology.left( e ) );
            if ( it1 != face2verts.end() )
            {
                for ( auto v : it1->second )
                    myverts.push_back( v );
                it1->second.clear();
            }
        }

        eliminateDoubleTrisAround( res.topology, newv );
        newFp.clear();
        for ( EdgeId e : orgRing( res.topology, newv ) )
        {
            auto & x = newFp.emplace_back();
            x.face = res.topology.left( e );
            x.plane = res.getPlane3d( x.face );
        }

        for ( auto v : myverts )
        {
            if ( v == topmostVert )
                continue;
            int bestFace = -1;
            double bestDist = 0;
            for ( int i = 0; i < newFp.size(); ++i )
            {
                const auto dist = newFp[i].plane.distance( Vector3d{ points[v] } );
                if ( dist > bestDist )
                {
                    bestFace = i;
                    bestDist = dist;
                }
            }
            if ( bestFace >= 0 )
            {
                newFp[bestFace].verts.push_back( v );
                newFp[bestFace].maxDist = std::max( newFp[bestFace].maxDist, bestDist );
            }
        }
        for ( auto & x : newFp )
        {
            queue.setValue( x.face, x.maxDist );
            if ( x.verts.empty() )
            {
                face2verts.erase( x.face );
                continue;
            }
            face2verts[x.face] = std::move( x.verts );
        }

/* uncomment for error checking
        if ( !findMultipleEdges( res.topology ).empty() )
            break;

        bool allConvex = true;
        for ( EdgeId e : undirectedEdges( res.topology ) )
        {
            auto v = res.dihedralAngleSin( e );
            if ( v < -1e-4 )
            {
                allConvex = false;
                break;
            }
        }
        if ( !allConvex )
            break;*/
    }

    return res;
}

Mesh makeConvexHull( const Mesh & in )
{
    return makeConvexHull( in.points, in.topology.getValidVerts() );
}

Mesh makeConvexHull( const PointCloud & in )
{
    return makeConvexHull( in.points, in.validPoints );
}

Contour2f makeConvexHull( Contour2f points )
{
    if ( points.size() < 2 )
        return points;

    auto minPointIt = std::min_element( points.begin(), points.end(), [] ( auto&& a, auto&& b )
    {
        return std::tie( a.y, a.x ) < std::tie( b.y, b.x );
    } );
    std::swap( *points.begin(), *minPointIt );
    const auto& minPoint = points.front();

    // sort points by polar angle and distance to the start point
    std::sort( points.begin() + 1, points.end(), [&] ( const Vector2f& a, const Vector2f& b )
    {
        const auto va = a - minPoint, vb = b - minPoint;
        if ( auto c = cross( va, vb ); c != 0.f )
            return c > 0.f;
        return va.lengthSq() > vb.lengthSq();
    } );

    size_t size = 2;
    for ( auto i = 2; i < points.size(); ++i )
    {
        if ( cross( points[i - 1] - minPoint, points[i - 0] - minPoint ) == 0.f )
        {
            assert( ( points[i - 1] - minPoint ).lengthSq() >= ( points[i - 0] - minPoint ).lengthSq() );
            continue;
        }

        const auto& p = points[i];
        while ( size >= 2 )
        {
            const auto& a = points[size - 2];
            const auto& b = points[size - 1];
            if ( cross( b - a, p - a ) > 0.f )
                break;
            size--;
        }
        points[size++] = p;
    }
    points.erase( points.begin() + size, points.end() );

    return points;
}

TEST( MRMesh, ConvexHull )
{
    Mesh torus = makeTorus( 1.0f, 0.3f, 16, 16 );
    Mesh discus = makeConvexHull( torus );
    EXPECT_EQ( discus.topology.numValidVerts(), 144 );
    EXPECT_EQ( discus.topology.numValidFaces(), 284 );
    EXPECT_EQ( discus.topology.lastNotLoneEdge(), EdgeId( 426 * 2 - 1 ) );
}

} //namespace MR
