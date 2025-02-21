#include "MRPolylineTrimWithPlane.h"
#include "MRPolyline.h"
#include "MRPolylineEdgeIterator.h"
#include "MRPlane3.h"
#include "MRGTest.h"

namespace MR
{

EdgeBitSet subdivideWithPlane( Polyline3& polyline, const Plane3f& plane, EdgeBitSet* new2Old, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback )
{
    if ( polyline.topology.numValidVerts() == 0 )
        return {};

    if ( new2Old )
        *new2Old = {};

    EdgeBitSet sectionEdges;
    const auto sectionPoints = extractSectionsFromPolyline( polyline, plane, 0.0f );
    for ( const auto& sectionPoint : sectionPoints )
    {
        const auto eNew = polyline.splitEdge( sectionPoint.e, polyline.edgePoint( sectionPoint.edgePointA() ) );
        sectionEdges.autoResizeSet( sectionPoint.e );
        if ( onEdgeSplitCallback )
            onEdgeSplitCallback( sectionPoint.e, eNew, sectionPoint.a );
    }

    EdgeBitSet visited = sectionEdges;
    for ( EdgeId e : sectionEdges )
    {
        if ( visited.test( e ) )
            continue;
        visited.set( e.sym() );

        EdgeId e0 = e;
        for ( ;; )
        {
            if ( !e0.valid() )
                break;

            visited.set( e0 );
            visited.set( e0.sym() );
            e0 = polyline.topology.next( e0.sym() );
        }
    }

    if ( new2Old )
        *new2Old = std::move( sectionEdges );

    return visited;
}

UndirectedEdgeBitSet fillPolylineLeft( const Polyline3& polyline, const EdgeBitSet& orgEdges, std::vector<VertPair>* cutSegments )
{
    const size_t numEdges = polyline.topology.lastNotLoneEdge().undirected() + 1;
    UndirectedEdgeBitSet res( numEdges );
    UndirectedEdgeBitSet visited( numEdges );

    if ( cutSegments )
        cutSegments->reserve( orgEdges.count() / 2 );

    for ( auto e : orgEdges )
    {
        if ( visited.test( e ) )
            continue;
            
        auto e0 = e;
        bool closed = false;
        for ( ;; )
        {
            if ( !e0.valid() )
                break;

            res.set( e0.undirected() );
            if ( orgEdges.test( e0.sym() ) )
            {
                visited.set( e0.sym() );
                closed = true;
                break;
            }
            if ( e0 == e0.sym() )
                break;
            e0 = polyline.topology.next( e0.sym() );
            if ( e0 == e )
                break;
        }
        if ( cutSegments && closed )
            cutSegments->push_back( { polyline.topology.org( e ), polyline.topology.org( e0.sym() ) } );
    }
    return res;
}

void trimWithPlane( Polyline3& polyline, const Plane3f& plane, const DividePolylineParameters& params )
{
    if ( polyline.points.empty() )
        return;

    EdgeBitSet newEdges;
    subdivideWithPlane( polyline, plane, &newEdges, params.onEdgeSplitCallback );
    if ( newEdges.empty() )
    {            
        if ( plane.distance( polyline.points.front() ) < 0 )
        {
            if ( params.otherPart )
                *params.otherPart = std::move( polyline );
            polyline = Polyline3{};
        }
        return;
    }

    std::vector<VertPair> cutSegments;
    const auto posEdges = fillPolylineLeft( polyline, newEdges, params.closeLineAfterCut ? &cutSegments : nullptr );
    Polyline3 res;
    VertMap vMap;
    res.addPartByMask( polyline, posEdges, &vMap, params.outEmap );
    if ( params.outVmap )
        *params.outVmap = std::move( vMap );

    if ( params.closeLineAfterCut )
        for ( const auto& segment : cutSegments )
            res.topology.makeEdge( vMap[segment.second], vMap[segment.first] );
                

    if ( params.otherPart )
    {
        const size_t numEdges = polyline.topology.lastNotLoneEdge().undirected() + 1;
        UndirectedEdgeBitSet otherPartEdges( numEdges );
        for ( auto ue : undirectedEdges( polyline.topology ) )
        {
            if ( !posEdges.test( ue ) )
                otherPartEdges.set( ue );
        }

        vMap.clear();
        params.otherPart->addPartByMask( polyline, otherPartEdges, &vMap, params.otherOutEmap );
        if ( params.otherOutVmap )
            *params.otherOutVmap = std::move( vMap );
        if ( params.closeLineAfterCut )
            for ( const auto& segment : cutSegments )
                params.otherPart->topology.makeEdge( vMap[segment.first], vMap[segment.second] );
    }
    polyline = std::move( res );
}

std::vector<EdgeSegment> extractSectionsFromPolyline( const Polyline3& polyline, const Plane3f& plane, float eps )
{
    std::vector<EdgeSegment> result;
    if ( polyline.topology.edgeSize() <= 0 )
        return {};

    Plane3f planePos( plane.n, plane.d + eps );
    Plane3f planeNeg( -plane.n, -plane.d + eps );

    struct PointPosition
    {
        Vector3f p;
        float distFromPosPlane = {};
        float distFromNegPlane = {};
    };
        
    for ( auto ue : undirectedEdges( polyline.topology ) )
    {
        const EdgeId e( ue );
            
        PointPosition p1{ .p = polyline.orgPnt( e ), .distFromPosPlane = planePos.distance( p1.p ), .distFromNegPlane = planeNeg.distance( p1.p ) };
        PointPosition p2{ .p = polyline.destPnt( e ), .distFromPosPlane = planePos.distance( p2.p ), .distFromNegPlane = planeNeg.distance( p2.p ) };

        bool isP1Between = p1.distFromNegPlane < 0 && p1.distFromPosPlane < 0;
        bool isP2Between = p2.distFromNegPlane < 0 && p2.distFromPosPlane < 0;

        EdgeSegment segment( e );

        if ( isP1Between && isP2Between )
        {
            result.push_back( segment );
        }
        else if ( isP1Between )
        {
            segment.b = p2.distFromPosPlane > 0 ? p1.distFromPosPlane / ( p1.distFromPosPlane - p2.distFromPosPlane )
                : p1.distFromNegPlane / ( p1.distFromNegPlane - p2.distFromNegPlane );
            result.push_back( segment );
        }
        else if ( isP2Between )
        {
            segment.a = p1.distFromPosPlane > 0 ? p1.distFromPosPlane / ( p1.distFromPosPlane - p2.distFromPosPlane )
                : p1.distFromNegPlane / ( p1.distFromNegPlane - p2.distFromNegPlane );
            result.push_back( segment );
        }
        else if ( p1.distFromPosPlane * p2.distFromPosPlane < 0 )
        {
            const float denom = ( p1.distFromPosPlane > 0 ) ? p1.distFromPosPlane + p2.distFromNegPlane + 2 * eps :
                                                                p1.distFromNegPlane + p2.distFromPosPlane + 2 * eps;
            if ( denom != 0 )
            {
                if ( p1.distFromPosPlane > 0 )
                {
                    segment.e = segment.e.sym();
                    segment.a = 1 - p1.distFromPosPlane / denom;
                    segment.b = p2.distFromNegPlane / denom;
                }
                else
                {
                    segment.a = p1.distFromNegPlane / denom;
                    segment.b = 1 - p2.distFromPosPlane / denom;
                }
            }
            result.push_back( segment );
        }
    }

    return result;
}


TEST( MRMesh, TrimPolylineWithPlane )
{
    Polyline3 polyline;

    std::vector<Vector3f> points = { {0.f, 0.f, 0.f}, {2.f, 2.f, 2.f}, {0.f, 4.f, 0.f} };
    polyline.addFromPoints( points.data(), points.size(), true );

    Plane3f plane( { 1.f, 0.f, 0.f }, 1.f );

    Polyline3 otherPart;
    DividePolylineParameters params;
    params.closeLineAfterCut = true;
    params.otherPart = &otherPart;
    trimWithPlane( polyline, plane, params );

    EXPECT_TRUE( polyline.topology.isClosed() );
    std::vector<VertId> pointsOnPlane( 2 );
    for ( VertId v : polyline.topology.getValidVerts() )
    {
        if ( polyline.points[v].x == 1.f )
            pointsOnPlane.push_back( v );
    }
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( polyline.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );

    EXPECT_TRUE( otherPart.topology.isClosed() );
    pointsOnPlane.clear();
    for ( VertId v : polyline.topology.getValidVerts() )
    {
        if ( otherPart.points[v].x == 1.f )
            pointsOnPlane.push_back( v );
    }
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( otherPart.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );



    polyline = {};
    polyline.addFromPoints( points.data(), points.size(), false );

    EXPECT_TRUE( polyline.topology.isClosed() );
    pointsOnPlane.clear();
    for ( VertId v : polyline.topology.getValidVerts() )
    {
        if ( polyline.points[v].x == 1.f )
            pointsOnPlane.push_back( v );
    }
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( polyline.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );

    EXPECT_FALSE( otherPart.topology.isClosed() );
    pointsOnPlane.clear();
    for ( VertId v : polyline.topology.getValidVerts() )
    {
        if ( otherPart.points[v].x == 1.f )
            pointsOnPlane.push_back( v );
    }
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( otherPart.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );
}


}
