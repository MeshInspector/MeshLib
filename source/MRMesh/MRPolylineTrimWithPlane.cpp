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
        Plane3f planeNeg( plane.n, plane.d - eps );
        
        for ( EdgeId e = EdgeId(0); e < edges.size(); ++e )
        {
            const auto& edgeRecord = edges[e];
            if ( !edgeRecord.next.valid() || !edgeRecord.org.valid() )
                continue;

            const auto p1 = polyline.orgPnt( e );
            const auto p2 = polyline.destPnt( e );

            float distPos1 = planePos.distance( p1 );
            float distPos2 = planePos.distance( p2 );
            float distNeg1 = planeNeg.distance( p1 );
            float distNeg2 = planeNeg.distance( p2 );

            bool isP1Between = ( distPos1 > 0 ) ^ ( distNeg1 > 0 );
            bool isP2Between = ( distPos2 > 0 ) ^ ( distNeg2 > 0 );



            if ( isP1Between && isP2Between )
                result.push_back( { EdgePoint( e, 0.0f ), EdgePoint( e, 1.0f ) } );

            if ( isP1Between )
            {
                result.push_back( { EdgePoint( e, 0.0f ), EdgePoint( e, distPos2 > 0 ? distPos1 / ( distPos1 - distPos2 ) : distNeg1 / ( distNeg1 - distNeg2 ) } ) );
            }

        }
    }
}