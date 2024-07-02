#include "MRPolylineTrimWithPlane.h"
#include "MRPolyline.h"
#include "MRPolylineEdgeIterator.h"
#include "MRPlane3.h"

namespace MR
{
    EdgeBitSet subdividePolylineWithPlane( Polyline3& polyline, const Plane3f& plane, std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback )
    {
        if ( polyline.topology.numValidVerts() == 0 )
            return {};

        EdgeBitSet res;
        const auto sectionPoints = extractSectionsFromPolyline( polyline, plane, 0.0f );
        for ( const auto& sectionPoint : sectionPoints )
        {
            const auto eNew = polyline.splitEdge( sectionPoint.e, polyline.edgePoint( sectionPoint.edgePointA() ) );            
            res.autoResizeSet( sectionPoint.e );
            if ( onEdgeSplitCallback )
                onEdgeSplitCallback( sectionPoint.e, eNew, sectionPoint.a );
        }

        return res;
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
            for ( ;;)
            {
                if ( !e0.valid() )
                    break;

                res.set( e0.undirected() );
                if ( orgEdges.test( e0.sym() ) )
                {
                    visited.set( e0.sym() );
                    break;
                }              
                e0 = polyline.topology.next( e0.sym() );
            }
            if ( cutSegments )
                cutSegments->push_back( { polyline.topology.org( e ), polyline.topology.org( e0.sym() ) } );
        }
        return res;
    }

    void dividePolylineWithPlane( Polyline3& polyline, const Plane3f& plane, const DividePolylineParameters& params )
    {
        if ( polyline.points.empty() )
            return;

        const auto newEdges = subdividePolylineWithPlane( polyline, plane, params.onEdgeSplitCallback );
        if ( newEdges.empty() )
        {            
            if ( plane.distance( polyline.points.front() ) < 0 )
            {
                if ( params.otherPart )
                    *params.otherPart = polyline;
                polyline = Polyline3{};
            }
            return;
        }

        std::vector<VertPair> cutSegments;
        const auto posEdges = fillPolylineLeft( polyline, newEdges, params.fillAfterCut ? &cutSegments : nullptr );
        Polyline3 res;
        VertMap vMap;
        res.addPartByMask( polyline, posEdges, &vMap, params.outEmap );
        if ( params.outVmap )
            *params.outVmap = vMap;

        if ( params.fillAfterCut )
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
                *params.otherOutVmap = vMap;
            if ( params.fillAfterCut )
                for ( const auto& segment : cutSegments )
                    params.otherPart->topology.makeEdge( vMap[segment.first], vMap[segment.second] );
        }
        polyline = res;
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
}