#include "MRPolylineTrimWithPlane.h"
#include "MRPolyline.h"
#include "MRPolylineEdgeIterator.h"
#include "MRPlane3.h"

namespace MR
{
    std::vector<std::vector<EdgePoint>> dividePolylineWithPlane( const Polyline3& polyline, const Plane3f& plane, std::vector<std::vector<EdgePoint>>* otherPartSegments )
    {
        std::vector<std::vector<EdgePoint>> result;
        const auto& edges = polyline.topology.edges();
        if ( edges.empty() )
            return result;

        float lastDistance = 0; // I gave you my heart...
        std::vector<EdgePoint>* currSegment = nullptr;
        std::vector<EdgePoint>* otherSegment = nullptr;

        if ( edges.front().org.valid() )
        {
            const auto p = polyline.points[edges.front().org];
            lastDistance = plane.distance( p );
            if ( lastDistance >= 0 )
            {
                result.emplace_back();
                currSegment = &result.back();
                currSegment->emplace_back( EdgeId(0), 0 );
            }
            else if ( otherPartSegments )
            {
                otherPartSegments->emplace_back();
                otherSegment = &otherPartSegments->back();
                otherSegment->emplace_back( EdgeId(0), 0 );
            }
        }        

        for ( EdgeId e = EdgeId(0); e < edges.size(); e+=2 )
        {
            const auto p = polyline.destPnt( e );
            const auto dist = plane.distance( p );
            if ( dist < 0 )
            {
                if ( currSegment )
                {
                    currSegment->emplace_back( e, lastDistance / ( lastDistance - dist ) );
                    currSegment = nullptr;
                }

                if ( otherPartSegments )
                {
                    if ( !otherSegment )
                    {
                        otherPartSegments->emplace_back();
                        otherSegment = &otherPartSegments->back();
                        otherSegment->emplace_back( e, lastDistance / ( lastDistance - dist ) );
                    }
                    else
                    {
                        otherSegment->emplace_back( e, 1 );
                    }                    
                }                
            }
            else
            {
                if ( !currSegment )
                {
                    result.emplace_back();
                    currSegment = &result.back();
                    currSegment->emplace_back( e, lastDistance / ( lastDistance - dist ) );
                }
                else
                {
                    currSegment->emplace_back( e, 1 );
                }

                if ( otherPartSegments && otherSegment )
                {
                    otherSegment->emplace_back( e, lastDistance / ( lastDistance - dist ) );
                    otherSegment = nullptr;
                }
            }

            lastDistance = dist;
        }

        return result;
    }

    Polyline3 dividePolylineWithPlane( const Polyline3& polyline, const Plane3f& plane, Polyline3* otherPart )
    {
        std::vector<std::vector<EdgePoint>> otherPartSections;
        const auto sections = dividePolylineWithPlane( polyline, plane, otherPart ? &otherPartSections : nullptr );
        Polyline3 res;        

        for ( const auto& section : sections )
        {
            std::vector<Vector3f> points;
            points.reserve( section.size() );
            for ( const auto& ep : section )
            {
                points.push_back( polyline.edgePoint( ep ) );
            }
            res.addFromPoints( points.data(), points.size() );
        }

        if ( otherPart )
        {
            for ( const auto& otherPartSection : otherPartSections )
            {
                std::vector<Vector3f> points;
                points.reserve( otherPartSection.size() );
                for ( const auto& ep : otherPartSection )
                {
                    points.push_back( polyline.edgePoint( ep ) );
                }
                otherPart->addFromPoints( points.data(), points.size() );
            }
        }

        return res;
    }

    std::vector<EdgeSegment> extractSectionsFromPolyline( const Polyline3& polyline, const Plane3f& plane, float eps )
    {
        std::vector<EdgeSegment> result;
        const auto& edges = polyline.topology.edges();
        if ( edges.empty() )
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
                if ( p1.distFromPosPlane > 0 )
                {
                    segment.e = segment.e.sym();
                    segment.a = 1 - p1.distFromPosPlane / denom;
                    segment.b = 1 - p2.distFromNegPlane / denom;
                }
                else
                {
                    segment.a = p1.distFromNegPlane / denom;
                    segment.b = p2.distFromPosPlane / denom;
                }
                result.push_back( segment );
            }
        }

        return result;
    }
}