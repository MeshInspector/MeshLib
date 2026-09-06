#include "MRCameraPointsTriangulation.h"
#include "MRMesh.h"
#include "MRTerrainTriangulation.h"
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

    VertCoords pixels( points.size() );
    ParallelFor( pixels, [&]( VertId v )
    {
        pixels[v] = project( points[v] );
    } );
    if ( !reportProgress( projectCb, 1.0f ) )
        return unexpectedOperationCanceled();

    VertCoords weldedPoints;
    if ( settings.weldPixels > 0 )
    {
        auto smallestMap = findSmallestCloseVertices( pixels, settings.weldPixels, nullptr, weldCb );
        if ( !smallestMap )
            return unexpectedOperationCanceled();

        VertCoords sums( points.size() );
        Vector<int, VertId> counts( points.size(), 0 );
        for ( VertId v( 0 ); v < points.size(); ++v )
        {
            const auto m = (*smallestMap)[v];
            sums[m] += points[v];
            ++counts[m];
        }
        pixels.clear();
        for ( VertId v( 0 ); v < points.size(); ++v )
        {
            if ( counts[v] == 0 )
                continue;
            const auto p = sums[v] / float( counts[v] );
            weldedPoints.push_back( p );
            pixels.push_back( project( p ) );
        }
    }
    else
        weldedPoints = points;

    if ( pixels.size() < 3 )
        return unexpected( "At least 3 distinct points are required" );

    auto res = terrainTriangulation( std::move( pixels.vec_ ), triCb );
    if ( !res )
        return res;
    Mesh & mesh = *res;
    mesh.points = std::move( weldedPoints );
    // counter-clockwise triangles in the image plane have normals along +Z, i.e. away from the camera
    mesh.topology.flipOrientation();
    return res;
}

} //namespace MR
