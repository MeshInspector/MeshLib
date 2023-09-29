#include "MRPolylineSubdivide.h"
#include "MRPolyline.h"
#include "MRPolylineEdgeIterator.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRWriter.h"
#include "MRGTest.h"
#include <queue>

namespace MR
{

namespace {

// computes normal of plane that approximately contains vectors a, b, c
Vector3f getPlaneNormal( const Vector3f & a, const Vector3f & b, const Vector3f & c )
{
    const auto planeNormal1 = cross( b,  a + c );
    const auto planeNormal2 = cross( b,  a - c );
    return planeNormal1.lengthSq() >= planeNormal2.lengthSq() ?
        planeNormal1 : planeNormal2;
}

inline int getPlaneNormal( const Vector2f &, const Vector2f &, const Vector2f & )
{
    return 0;
}

// rotation around planeNormal on 90 degrees
inline Vector3f getNormalInPlane( const Vector3f & planeNormal, const Vector3f & a )
{
    return cross( planeNormal, a );
}

// rotation on 90 degrees in 2D plane
inline Vector2f getNormalInPlane( int, const Vector2f & a )
{
    return Vector2f{ -a.y, a.x };
}

} //anonymous namespace

struct EdgeLength
{
    UndirectedEdgeId edge;
    float lenSq = 0; // at the moment the edge was put in the queue

    EdgeLength() = default;
    EdgeLength( UndirectedEdgeId edge, float lenSq ) : edge( edge ), lenSq( lenSq ) {}
};

inline bool operator < ( const EdgeLength & a, const EdgeLength & b )
{
    return std::tie( a.lenSq, a.edge ) < std::tie( b.lenSq, b.edge );
}

template<typename V>
int subdividePolylineT( Polyline<V> & polyline, const PolylineSubdivideSettings & settings )
{
    MR_TIMER;

    const float maxEdgeLenSq = sqr( settings.maxEdgeLen );
    std::priority_queue<EdgeLength> queue;

    // region is changed during subdivision,
    // so if it has invalid vertices (they can become valid later) some collisions can occur
    // better to filter valid vertices in first step
    if ( settings.region )
        *settings.region &= polyline.topology.getValidVerts();

    auto addInQueue = [&]( UndirectedEdgeId e )
    {
        const bool canSubdivide = !settings.region ||
            ( settings.region->test( polyline.topology.org( e ) ) && settings.region->test( polyline.topology.dest( e ) ) );
        if ( !canSubdivide )
            return;
        float lenSq = polyline.edgeLengthSq( e );
        if ( lenSq < maxEdgeLenSq )
            return;
        queue.emplace( e, lenSq );
    };

    for ( UndirectedEdgeId e : undirectedEdges( polyline.topology ) )
    {
        addInQueue( e );
    }

    if ( settings.progressCallback && !settings.progressCallback( 0.25f ) )
        return 0;

    MR_WRITER( polyline );

    int splitsDone = 0;
    int lastProgressSplitsDone = 0;
    while ( splitsDone < settings.maxEdgeSplits && !queue.empty() )
    {
        if ( settings.progressCallback && splitsDone >= 1000 + lastProgressSplitsDone ) 
        {
            if ( !settings.progressCallback( 0.25f + 0.75f * splitsDone / settings.maxEdgeSplits ) )
                return splitsDone;
            lastProgressSplitsDone = splitsDone;
        }
        auto el = queue.top();
        queue.pop();
        if ( el.lenSq != polyline.edgeLengthSq( el.edge ) )
            continue; // outdated record in the queue
        EdgeId e = el.edge;

        auto newVertPos = polyline.edgeCenter( e );
        if ( settings.useCurvature )
        {
            const auto e1 = polyline.topology.next( e );
            const auto e2 = polyline.topology.next( e.sym() );
            if ( e != e1 && e.sym() != e2 )
            {
                const auto p1 = polyline.destPnt( e1 );
                const auto po = polyline.orgPnt( e );
                const auto pd = polyline.destPnt( e );
                const auto p2 = polyline.destPnt( e2 );

                const auto a = po - p1;
                const auto b = pd - po;
                const auto c = p2 - pd;
                const auto planeNormal = getPlaneNormal( a, b, c );

                const auto nod = getNormalInPlane( planeNormal, b ).normalized();
                const auto no = ( nod + getNormalInPlane( planeNormal, a ).normalized() ).normalized();
                const auto nd = ( nod + getNormalInPlane( planeNormal, c ).normalized() ).normalized();
                const float sign = dot( pd - po, nd - no ) >= 0 ? 1.f : -1.f;
                newVertPos = 0.5f * ( po + pd + sign * std::tan( angle( no, nd ) / 4 ) * b.length() * ( no + nd ).normalized()  );
            }
        }
        const auto e1 = polyline.splitEdge( e, newVertPos );
        const auto newVertId = polyline.topology.org( e );

        if ( settings.region )
            settings.region->autoResizeSet( newVertId );
        if ( settings.newVerts )
            settings.newVerts->autoResizeSet( newVertId );
        if ( settings.onVertCreated )
            settings.onVertCreated( newVertId );
        if ( settings.onEdgeSplit )
            settings.onEdgeSplit( e1, e );
        ++splitsDone;
        assert( e != polyline.topology.next( e ) );
        addInQueue( e.undirected() );
        addInQueue( polyline.topology.next( e ).undirected() );
    }

    return splitsDone;
}

int subdividePolyline( Polyline2 & polyline, const PolylineSubdivideSettings & settings )
{
    return subdividePolylineT( polyline, settings );
}

int subdividePolyline( Polyline3 & polyline, const PolylineSubdivideSettings & settings )
{
    return subdividePolylineT( polyline, settings );
}

TEST(MRMesh, SubdividePolyline)
{
    Contour2f cont;
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 1.f ) );
    cont.push_back( Vector2f( 0.f, 0.f ) );
    Polyline2 polyline( { cont } );

    PolylineSubdivideSettings settings;
    settings.maxEdgeLen = 0.3f;
    settings.maxEdgeSplits = 1000;
    int splitsDone = subdividePolyline( polyline, settings );
    EXPECT_TRUE( splitsDone > 11 && splitsDone < 15 );
}

} // namespace MR
