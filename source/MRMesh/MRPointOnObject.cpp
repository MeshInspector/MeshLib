#include "MRPointOnObject.h"
#include "MRObjectMeshHolder.h"
#include "MRObjectPointsHolder.h"
#include "MRObjectLinesHolder.h"
#include "MRPointCloud.h"
#include "MRMeshTriPoint.h"
#include "MREdgePoint.h"
#include "MRPolyline.h"
#include "MRMesh.h"

namespace MR
{

PickedPoint pointOnObjectToPickedPoint( const VisualObject* object, const PointOnObject& pos )
{
    if ( auto* objMesh = dynamic_cast< const ObjectMeshHolder* >( object ) )
        return objMesh->mesh()->toTriPoint( pos );

    if ( dynamic_cast< const ObjectPointsHolder* >( object ) )
        return pos.vert;

    if ( auto* objLines  = dynamic_cast< const ObjectLinesHolder* >( object ) )
        return objLines->polyline()->toEdgePoint( EdgeId( pos.uedge ), pos.point );

    assert( false );
    return {};
}

std::optional<Vector3f> getPickedPointPosition( const VisualObject& object, const PickedPoint& point )
{
    return std::visit( overloaded{
        []( const std::monostate& ) -> std::optional<Vector3f>
        {
            return {};
        },
        [&object]( const MeshTriPoint& triPoint ) -> std::optional<Vector3f>
        {
            if ( auto objMesh = dynamic_cast< const ObjectMeshHolder* >( &object ) )
            {
                if ( const auto& mesh = objMesh->mesh() )
                {
                    const auto & topology = mesh->topology;
                    if ( topology.hasEdge( triPoint.e ) )
                    {
                        if ( triPoint.bary.b == 0 || topology.left( triPoint.e ) )
                            return mesh->triPoint( triPoint );
                    }
                }
            }
            return {};
        },
        [&object]( const EdgePoint& edgePoint ) -> std::optional<Vector3f>
        {
            if ( auto objLines = dynamic_cast< const ObjectLinesHolder* >( &object ) )
            {
                if ( const auto& polyline = objLines->polyline() )
                {
                    const auto & topology = polyline->topology;
                    if ( topology.hasEdge( edgePoint.e ) )
                        return objLines->polyline()->edgePoint( edgePoint );
                }
            }
            return {};
        },
        [&object]( VertId vertId ) -> std::optional<Vector3f>
        {
            if ( auto objPoints = dynamic_cast< const ObjectPointsHolder* >( &object ) )
            {
                if ( const auto& pointCloud = objPoints->pointCloud() )
                {
                    if ( pointCloud->validPoints.test( vertId ) )
                        return pointCloud->points[vertId];
                }
            }
            return {};
        }
    }, point );
}

Vector3f pickedPointToVector3( const VisualObject* object, const PickedPoint& point )
{
    auto opt = getPickedPointPosition( *object, point );
    if ( opt )
        return *opt;
    assert( false );
    return {};
}

bool isPickedPointValid( const VisualObject* object, const PickedPoint& point )
{
    return getPickedPointPosition( *object, point ).has_value();
}

} //namespace MR
