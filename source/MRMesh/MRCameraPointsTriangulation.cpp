#include "MRCameraPointsTriangulation.h"
#include "MRMesh.h"
#include "MRTerrainTriangulation.h"
#include "MRCloseVertices.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRTimer.h"

namespace MR
{

Expected<Mesh> triangulateCameraPoints( const VertCoords & points, const CameraPointsTriangulationSettings & settings )
{
    MR_TIMER;

    if ( points.size() < 3 )
        return unexpected( "At least 3 points are required" );

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
    if ( !reportProgress( settings.cb, 0.05f ) )
        return unexpectedOperationCanceled();

    VertCoords weldedPoints;
    if ( settings.weldPixels > 0 )
    {
        auto smallestMap = findSmallestCloseVertices( pixels, settings.weldPixels, nullptr, subprogress( settings.cb, 0.05f, 0.2f ) );
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

    auto res = terrainTriangulation( std::move( pixels.vec_ ), subprogress( settings.cb, 0.2f, 0.9f ) );
    if ( !res )
        return res;
    Mesh & mesh = *res;
    mesh.points = std::move( weldedPoints );
    // counter-clockwise triangles in the image plane have normals along +Z, i.e. away from the camera
    mesh.topology.flipOrientation();

    if ( settings.maxEdgeLength > 0 )
    {
        const float maxLenSq = sqr( settings.maxEdgeLength );
        FaceBitSet longFaces( mesh.topology.faceSize() );
        BitSetParallelFor( mesh.topology.getValidFaces(), [&]( FaceId f )
        {
            for ( EdgeId e : leftRing( mesh.topology, f ) )
            {
                if ( mesh.edgeLengthSq( e.undirected() ) > maxLenSq )
                {
                    longFaces.set( f );
                    break;
                }
            }
        } );
        mesh.topology.deleteFaces( longFaces );
        mesh.pack();
    }

    if ( !reportProgress( settings.cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

} //namespace MR
