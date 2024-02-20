#include "MRPolylineTrimWithPlane.h"
#include "MRPolyline.h"
#include "MRPlane3.h"

namespace MR
{
    std::vector<std::pair<EdgePoint, EdgePoint>> trimPolylineWithPlane( const Polyline3& polyline, const Plane3f& plane, float eps )
    {
        std::vector<std::pair<EdgePoint, EdgePoint>> result;
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
        
        for ( EdgeId e = EdgeId(0); e < edges.size(); ++e )
        {
            const auto& edgeRecord = edges[e];
            if ( !edgeRecord.next.valid() || !edgeRecord.org.valid() )
                continue;
            
            PointPosition p1{ .p = polyline.orgPnt( e ), .distFromPosPlane = planePos.distance( p1.p ), .distFromNegPlane = planeNeg.distance( p1.p ) };
            PointPosition p2{ .p = polyline.destPnt( e ), .distFromPosPlane = planePos.distance( p2.p ), .distFromNegPlane = planeNeg.distance( p2.p ) };

            bool isP1Between = p1.distFromNegPlane < 0 && p1.distFromPosPlane < 0;
            bool isP2Between = p2.distFromNegPlane < 0 && p2.distFromPosPlane < 0;

            std::pair<EdgePoint, EdgePoint> segment( EdgePoint( e, 0.0f ), EdgePoint( e, 0.0f ) );

            if ( isP1Between && isP2Between )
                segment.second.a = 1.0f;

            if ( isP1Between )
            {
                segment.second.a = p2.distFromPosPlane > 0 ? p1.distFromPosPlane / ( p1.distFromPosPlane - p2.distFromPosPlane )
                                                           : p1.distFromNegPlane / ( p1.distFromNegPlane - p2.distFromNegPlane );
                result.push_back( segment );
            }

            if ( isP2Between )
            {
                segment.first.a = p1.distFromPosPlane > 0 ? p1.distFromPosPlane / ( p1.distFromPosPlane - p2.distFromPosPlane )
                                                          : p1.distFromNegPlane / ( p1.distFromNegPlane - p2.distFromNegPlane );
                segment.second.a = 1.0f;
                result.push_back( segment );
            }

            if ( p1.distFromPosPlane * p2.distFromPosPlane < 0 )
            {
                if ( p1.distFromPosPlane > 0 )
                {
                    const float det = 2 * p2.distFromNegPlane * ( p1.distFromPosPlane + eps );
                    segment.first.a = p1.distFromPosPlane * p1.distFromPosPlane / det;
                    segment.second.a = segment.first.a + 2 * eps / det;
                }
                else
                {
                    const float det = 2 * p2.distFromPosPlane * ( p1.distFromNegPlane + eps );
                    segment.first.a = p1.distFromNegPlane * p1.distFromNegPlane / det;
                    segment.second.a = segment.first.a + 2 * eps / det;
                }
                result.push_back( segment );
            }
        }

        return result;
    }
}