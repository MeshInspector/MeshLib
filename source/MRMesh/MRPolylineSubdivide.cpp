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

struct EdgeLength
{
    UndirectedEdgeId edge;
    float lenSq = 0; // at the moment the edge was put in the queue

    EdgeLength() = default;
    EdgeLength( UndirectedEdgeId edge, float lenSq ) : edge( edge ), lenSq( lenSq ) {}

    auto asPair() const { return std::make_pair( lenSq, edge ); }
};

inline bool operator < ( const EdgeLength & a, const EdgeLength & b )
{
    return a.asPair() < b.asPair();
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
        auto newVertId = polyline.splitEdge( e );
        if ( settings.region )
            settings.region->autoResizeSet( newVertId );
        if ( settings.newVerts )
            settings.newVerts->autoResizeSet( newVertId );
        if ( settings.onVertCreated )
            settings.onVertCreated( newVertId );
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
