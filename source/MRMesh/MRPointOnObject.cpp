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
    const ObjectMeshHolder* objMesh = dynamic_cast< const ObjectMeshHolder* >( object );
    if ( objMesh )
    {
        return objMesh->mesh()->toTriPoint( pos );
    }
    const ObjectPointsHolder* objPoints = dynamic_cast< const ObjectPointsHolder* >( object );
    if ( objPoints )
    {
        return pos.vert;
    }

    const ObjectLinesHolder* objLines = dynamic_cast< const ObjectLinesHolder* >( object );
    if ( objLines )
    {
        return objLines->polyline()->toEdgePoint( MR::EdgeId( pos.uedge ), pos.point );
    }

    return -1;
}

MR::Vector3f pickedPointToVector3( const VisualObject* object, const PickedPoint& point )
{

    if ( const MeshTriPoint* triPoint = std::get_if<MeshTriPoint>( &point ) )
    {
        const auto objMesh = dynamic_cast< const ObjectMeshHolder* >( object );
        assert( objMesh );
        if ( objMesh )
        {
            return objMesh->mesh()->triPoint( *triPoint );
        }
    }
    else if ( const VertId* vertId = std::get_if<VertId>( &point ) )
    {
        const auto objPoints = dynamic_cast< const ObjectPointsHolder* >( object );
        assert( objPoints );
        if ( objPoints )
        {
            return objPoints->pointCloud()->points[*vertId];
        }
    }
    else if ( const EdgePoint* edgePoint = std::get_if<EdgePoint>( &point ) )
    {
        const auto objLines = dynamic_cast< const ObjectLinesHolder* >( object );
        assert( objLines );
        if ( objLines )
        {
            return objLines->polyline()->edgePoint( *edgePoint );
        }
    }
    else if ( std::get_if<int>( &point ) )
    {
        assert( false ); // not valid object for pick points
        return {};
    }

    assert( false ); // not supported type in PickedPoint variant
    return {};
}

bool isPickedPointValid( const VisualObject* object, const PickedPoint& point )
{
    return std::visit( overloaded {
        [object] ( const MeshTriPoint& mtp )
        {
            if ( const auto* objMesh = dynamic_cast<const ObjectMeshHolder*>( object ) )
            {
                const auto& topology = objMesh->mesh()->topology;
                return topology.hasEdge( mtp.e ) && topology.hasFace( topology.left( mtp.e ) );
            }
            return false;
        },
        [object] ( const EdgePoint& ep )
        {
            if ( const auto* objLines = dynamic_cast<const ObjectLinesHolder*>( object ) )
            {
                const auto& topology = objLines->polyline()->topology;
                return topology.hasEdge( ep.e );
            }
            return false;
        },
        [object] ( const VertId& v )
        {
            if ( const auto* objPoints = dynamic_cast<const ObjectPointsHolder*>( object ) )
                return objPoints->pointCloud()->validPoints.test( v );
            return false;
        },
        [] ( int )
        {
            return false;
        }
    }, point );
}

} //namespace MR
