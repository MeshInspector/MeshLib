#include "MRPointOnObject.h"
#include "MRObjectMeshHolder.h"
#include "MRObjectPointsHolder.h"
#include "MRObjectLinesHolder.h"
#include "MRPointCloud.h"
#include "MRMeshTriPoint.h"
#include "MREdgePoint.h"
#include "MRPolyline.h"
#include "MRMesh.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

PickedPoint pointOnObjectToPickedPoint( const VisualObject* object, const PointOnObject& pos )
{
    if ( auto* objMesh = dynamic_cast< const ObjectMeshHolder* >( object ) )
    {
        const auto & mesh = objMesh->mesh();
        // toTriPoint() indexes edgePerFace_ by the face, so an out-of-range one reads out of bounds
        if ( !mesh || !pos.face.valid() || !mesh->topology.hasFace( pos.face ) )
        {
            spdlog::warn( "pointOnObjectToPickedPoint: not a valid mesh pick: face={}, faceSize={}",
                int( pos.face ), mesh ? mesh->topology.faceSize() : 0 );
            return {};
        }
        return mesh->toTriPoint( pos );
    }

    if ( auto* objPoints = dynamic_cast< const ObjectPointsHolder* >( object ) )
    {
        const auto & cloud = objPoints->pointCloud();
        if ( !cloud || !pos.vert.valid() || !cloud->validPoints.test( pos.vert ) )
        {
            spdlog::warn( "pointOnObjectToPickedPoint: not a valid point pick: vert={}, numPoints={}",
                int( pos.vert ), cloud ? cloud->points.size() : 0 );
            return {};
        }
        return pos.vert;
    }

    if ( auto* objLines  = dynamic_cast< const ObjectLinesHolder* >( object ) )
    {
        const auto & polyline = objLines->polyline();
        const EdgeId e( pos.uedge );
        if ( !polyline || !e.valid() || !polyline->topology.hasEdge( e ) )
        {
            spdlog::warn( "pointOnObjectToPickedPoint: not a valid polyline pick: uedge={}, edgeSize={}",
                int( pos.uedge ), polyline ? polyline->topology.edgeSize() : 0 );
            return {};
        }
        return polyline->toEdgePoint( e, pos.point );
    }

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

std::optional<Vector3f> getPickedPointNormal( const VisualObject& object, const PickedPoint& point, bool interpolated )
{
    return std::visit( overloaded{
        []( const std::monostate& ) -> std::optional<Vector3f>
        {
            return {};
        },
        [&object,interpolated] ( const MeshTriPoint& triPoint ) -> std::optional<Vector3f>
        {
            if ( auto objMesh = dynamic_cast< const ObjectMeshHolder* >( &object ) )
            {
                if ( const auto& mesh = objMesh->mesh() )
                {
                    const auto & topology = mesh->topology;
                    if ( topology.hasEdge( triPoint.e ) )
                    {
                        if ( triPoint.bary.b == 0 || topology.left( triPoint.e ) )
                            return interpolated ? mesh->normal( triPoint ) : mesh->pseudonormal( triPoint );
                    }
                }
            }
            return {};
        },
        []( const EdgePoint& ) -> std::optional<Vector3f>
        {
            return {};
        },
        [&object]( VertId vertId ) -> std::optional<Vector3f>
        {
            if ( auto objPoints = dynamic_cast< const ObjectPointsHolder* >( &object ) )
            {
                if ( const auto& pointCloud = objPoints->pointCloud() )
                {
                    if ( vertId < pointCloud->normals.size() && pointCloud->validPoints.test( vertId ) )
                        return pointCloud->normals[vertId];
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
