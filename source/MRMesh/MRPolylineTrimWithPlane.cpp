#include "MRPolylineTrimWithPlane.h"
#include "MRPolyline.h"
#include "MRPolylineEdgeIterator.h"
#include "MRPlane3.h"

namespace MR
{

UndirectedEdgeBitSet subdivideWithPlane( Polyline3& polyline, const Plane3f& plane, EdgeBitSet* newPositiveEdges, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback )
{
    if ( polyline.topology.numValidVerts() == 0 )
        return {};

    UndirectedEdgeBitSet result;
    const auto sectionPoints = extractSectionsFromPolyline( polyline, plane, 0.0f, &result );
    for ( const auto& sectionPoint : sectionPoints )
    {
        const auto eNew = polyline.splitEdge( sectionPoint.e, polyline.edgePoint( sectionPoint.edgePointA() ) );
        result.autoResizeSet( sectionPoint.e.undirected() );
        if ( newPositiveEdges )
            newPositiveEdges->autoResizeSet( sectionPoint.e );
        if ( onEdgeSplitCallback )
            onEdgeSplitCallback( sectionPoint.e, eNew, sectionPoint.a );
    }

    return result;
}

UndirectedEdgeBitSet subdividePolylineWithPlane( Polyline3& polyline, const Plane3f& plane, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback /*= nullptr */ )
{
    return subdivideWithPlane( polyline, plane, nullptr, onEdgeSplitCallback );
}

std::vector<VertPair> findSegmentEndVertices( const Polyline3& polyline, const EdgeBitSet& orgEdges )
{
    const size_t numEdges = polyline.topology.lastNotLoneEdge().undirected() + 1;
    UndirectedEdgeBitSet visited( numEdges );

    std::vector<VertPair> result;
    result.reserve( orgEdges.count() / 2 );

    for ( auto e : orgEdges )
    {
        if ( visited.test( e ) )
            continue;
            
        auto e0 = e;
        for ( ;; )
        {
            if ( !e0.valid() )
                break;

            e0 = e0.sym();
            if ( orgEdges.test( e0 ) )
            {
                visited.set( e0 );
                break;
            }
            e0 = polyline.topology.next( e0 );
            if ( e0 == e )
                break;
        }
        if ( e0.valid() && e0 != e )
            result.push_back( { polyline.topology.org( e ), polyline.topology.org( e0 ) } );
    }
    return result;
}

void trimWithPlane( Polyline3& polyline, const Plane3f& plane, const DividePolylineParameters& params )
{
    if ( polyline.points.empty() )
        return;

    EdgeBitSet newEdges;
    const auto posEdges = subdivideWithPlane( polyline, plane, &newEdges, params.onEdgeSplitCallback );
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
    if ( params.closeLineAfterCut )
        cutSegments = findSegmentEndVertices( polyline, newEdges );
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

void dividePolylineWithPlane( Polyline3& polyline, const Plane3f& plane, const DividePolylineParameters& params /*= {} */ )
{
    trimWithPlane( polyline, plane, params );
}

std::vector<EdgeSegment> extractSectionsFromPolyline( const Polyline3& polyline, const Plane3f& plane, float eps, UndirectedEdgeBitSet* positiveEdges )
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
    
    if ( positiveEdges && positiveEdges->size() < polyline.topology.undirectedEdgeSize() )
        positiveEdges->resize( polyline.topology.undirectedEdgeSize() );

    for ( auto ue : undirectedEdges( polyline.topology ) )
    {
        const EdgeId e( ue );
            
        PointPosition p1{ .p = polyline.orgPnt( e ), .distFromPosPlane = planePos.distance( p1.p ), .distFromNegPlane = planeNeg.distance( p1.p ) };
        PointPosition p2{ .p = polyline.destPnt( e ), .distFromPosPlane = planePos.distance( p2.p ), .distFromNegPlane = planeNeg.distance( p2.p ) };

        bool isP1Between = p1.distFromNegPlane <= 0 && p1.distFromPosPlane <= 0;
        bool isP2Between = p2.distFromNegPlane <= 0 && p2.distFromPosPlane <= 0;

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
                    segment.a = p2.distFromNegPlane / denom;
                    segment.b = 1 - p1.distFromPosPlane / denom;
                }
                else
                {
                    segment.a = p1.distFromNegPlane / denom;
                    segment.b = 1 - p2.distFromPosPlane / denom;
                }
            }
            result.push_back( segment );
        }
        else if ( positiveEdges && p1.distFromPosPlane > 0.f && p2.distFromPosPlane > 0.f )
        {
            positiveEdges->set( ue );
        }
    }

    return result;
}

}
