#include "MRCameraPointsTriangulation.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRDelaunayTriangulationXY.h"
#include "MRCloseVertices.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

Expected<Mesh> triangulateCameraPoints( const VertCoords & points, const CameraPointsTriangulationSettings & settings, const ProgressCallback & cb )
{
    MR_TIMER;

    if ( points.size() < 3 )
        return unexpected( "At least 3 points are required" );

    auto [projectCb, weldCb, triCb] = splitProgress( cb, 0.05f, 0.2f );

    auto project = [&K = settings.intrinsics]( const Vector3f & p )
    {
        const auto q = K * p;
        return Vector3f( q.x / q.z, q.y / q.z, 0 );
    };

    // image-plane positions with zero third coordinate; valid points are the ones to triangulate
    PointCloud pixels;
    pixels.points.resize( points.size() );
    ParallelFor( pixels.points, [&]( VertId v )
    {
        pixels.points[v] = project( points[v] );
    } );
    pixels.validPoints.resize( points.size(), true );
    if ( !reportProgress( projectCb, 1.0f ) )
        return unexpectedOperationCanceled();

    VertCoords meshPoints = points;
    if ( settings.weldPixels > 0 )
    {
        auto smallestMap = findSmallestCloseVertices( pixels.points, settings.weldPixels, nullptr, weldCb );
        if ( !smallestMap )
            return unexpectedOperationCanceled();

        VertCoords sums( points.size() );
        Vector<int, VertId> counts( points.size(), 0 );
        for ( VertId v( 0 ); v < points.size(); ++v )
        {
            const auto m = (*smallestMap)[v];
            sums[m] += points[v];
            ++counts[m];
            if ( m != v )
                pixels.validPoints.reset( v );
        }
        for ( VertId v : pixels.validPoints )
        {
            if ( counts[v] <= 1 )
                continue;
            meshPoints[v] = sums[v] / float( counts[v] );
            pixels.points[v] = project( meshPoints[v] );
        }
        if ( settings.outSmallestMap )
            *settings.outSmallestMap = std::move( *smallestMap );
    }

    if ( pixels.validPoints.count() < 3 )
        return unexpected( "At least 3 distinct points are required" );

    auto res = delaunayTriangulationXY( std::move( pixels ), triCb );
    if ( !res )
        return res;
    res->points = std::move( meshPoints );
    // counter-clockwise triangles in the image plane have normals along +Z, i.e. away from the camera
    res->topology.flipOrientation();
    return res;
}

} //namespace MR
