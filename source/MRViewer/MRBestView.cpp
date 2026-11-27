#include "MRBestView.h"
#include <MRPch/MRTBB.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRObjectMesh.h>

namespace MR
{

// Calculate average normal for all valid points
static Vector3f averageNormal( const PointCloud &pointsCloud )
{
    Vector3d sum = parallel_deterministic_reduce(
        tbb::blocked_range( 0_v, VertId{ pointsCloud.points.size() }, 1024 ), Vector3d{},
        [&] ( const auto& range, Vector3d curr )
        {
            for ( VertId v = range.begin(); v < range.end(); ++v )
                if ( pointsCloud.validPoints.test( v ) )
                    curr += Vector3d{ pointsCloud.normals[v] };
            return curr;
        },
        std::plus<Vector3d>() );
    return Vector3f( sum );
}

// Calculate average normalized direction from center for all valid points
static Vector3f averageDirection( const PointCloud& pointsCloud )
{
    Vector3d center = Vector3d( pointsCloud.findCenterFromBBox() );
    Vector3d sum = parallel_deterministic_reduce(
        tbb::blocked_range( 0_v, VertId{ pointsCloud.points.size() }, 1024 ), Vector3d{},
        [&] ( const auto& range, Vector3d curr )
        {
            for ( VertId v = range.begin(); v < range.end(); ++v )
                if ( pointsCloud.validPoints.test( v ) )
                    curr += ( Vector3d{ pointsCloud.points[v] } - center ).normalized();
            return curr;
        },
        std::plus<Vector3d>() );
    return Vector3f( sum );
}

Vector3f findAverageSurfaceVector( const VisualObject& obj )
{
    Vector3f r{};
    if ( const ObjectPoints* objPoints = obj.asType<ObjectPoints>() )
    {
        if ( const PointCloud* cloud = objPoints->pointCloudPtr() )
        {
            if ( cloud->hasNormals() )
                r = averageNormal( *cloud );
            else
                r = averageDirection( *cloud );
        }
    }
    else if ( const ObjectMesh* objMesh = obj.asType<ObjectMesh>() )
    {
        if ( const Mesh* mesh = objMesh->meshPtr() )
            r = Vector3f( mesh->dirArea() );
    }
    return r.normalized();
}

}
