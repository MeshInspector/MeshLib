#include "MRCameraPointsTriangulation.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRDelaunayTriangulationXY.h"
#include "MRCloseVertices.h"
#include "MRMeshMath.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

static Expected<Mesh> triangulateCameraPoints( VertCoords points, const VertBitSet * validPoints, const CameraPointsTriangulationSettings & settings, const ProgressCallback & cb )
{
    MR_TIMER;

    auto [projectCb, weldCb, triCb] = splitProgress( cb, 0.05f, 0.2f );

    // y is mirrored so that counter-clockwise triangles in the image plane face the camera, which looks along +Z
    auto project = [&K = settings.intrinsics]( const Vector3f & p )
    {
        const auto q = K * p;
        return Vector3f( q.x / q.z, -q.y / q.z, 1 );
    };

    // image-plane positions; valid points are the ones to triangulate
    PointCloud pixels;
    pixels.points.resize( points.size() );
    if ( validPoints )
        pixels.validPoints = *validPoints;
    else
        pixels.validPoints.resize( points.size(), true );
    BitSetParallelFor( pixels.validPoints, [&]( VertId v )
    {
        pixels.points[v] = project( points[v] );
    } );
    if ( !reportProgress( projectCb, 1.0f ) )
        return unexpectedOperationCanceled();

    if ( settings.weldPixels > 0 )
    {
        auto smallestMap = findSmallestCloseVertices( pixels.points, settings.weldPixels, &pixels.validPoints, weldCb );
        if ( !smallestMap )
            return unexpectedOperationCanceled();

        VertCoords sums( points.size() );
        Vector<int, VertId> counts( points.size(), 0 );
        for ( VertId v : pixels.validPoints )
        {
            const auto m = (*smallestMap)[v];
            sums[m] += points[v];
            ++counts[m];
        }
        for ( VertId v : pixels.validPoints )
        {
            if ( (*smallestMap)[v] != v )
                pixels.validPoints.reset( v );
            else if ( counts[v] > 1 )
            {
                points[v] = sums[v] / float( counts[v] );
                pixels.points[v] = project( points[v] );
            }
        }
        if ( settings.outSmallestMap )
            *settings.outSmallestMap = std::move( *smallestMap );
    }

    auto res = delaunayTriangulationXY( std::move( pixels ), triCb );
    if ( !res )
        return res;
    if ( settings.outProjectedPoints )
        *settings.outProjectedPoints = std::move( res->points );
    res->points = std::move( points );
    return res;
}

Expected<Mesh> triangulateCameraPoints( const VertCoords & points, const CameraPointsTriangulationSettings & settings, const ProgressCallback & cb )
{
    return triangulateCameraPoints( points, nullptr, settings, cb );
}

void smoothCameraMeshDepth( Mesh & mesh, const SmoothCameraMeshDepthSettings & settings )
{
    MR_TIMER;
    const auto & verts = mesh.topology.getVertIds( settings.region );
    VertScalars depth( mesh.points.size() );
    // the mesh projected in the image plane (normalized coordinates), where edge weights and vertex areas are computed
    VertCoords projected( mesh.points.size() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        const auto & p = mesh.points[v];
        depth[v] = p.z;
        projected[v] = Vector3f( p.x / p.z, -p.y / p.z, 1 );
    } );

    VertScalars stabilizers( mesh.points.size(), settings.innerStabilizer );
    if ( settings.innerStabilizerType != AreaStabilizer::Uniform )
    {
        BitSetParallelFor( verts, [&]( VertId v )
        {
            const auto a = dblArea( mesh.topology, projected, v );
            stabilizers[v] = settings.innerStabilizerType == AreaStabilizer::Area ? a : sqr( a );
        } );
        double sum = 0;
        int n = 0;
        for ( auto v : verts )
        {
            if ( mesh.topology.isBdVertex( v ) )
                continue;
            sum += stabilizers[v];
            ++n;
        }
        const float k = sum > 0 ? float( settings.innerStabilizer * n / sum ) : 0;
        BitSetParallelFor( verts, [&]( VertId v )
        {
            stabilizers[v] *= k;
        } );
    }
    for ( auto v : verts )
        if ( mesh.topology.isBdVertex( v ) )
            stabilizers[v] = settings.bdStabilizer;

    InterpolateScalarsParams params
    {
        .region = settings.region,
        .edgeWeights = settings.edgeWeights,
        .vertStabilizers = [&stabilizers]( VertId v ) { return stabilizers[v]; }
    };
    interpolateScalarsSmoothly( mesh.topology, projected, depth, params );
    BitSetParallelFor( verts, [&]( VertId v )
    {
        auto & p = mesh.points[v];
        if ( p.z > 0 && depth[v] > 0 )
            p *= depth[v] / p.z;
    } );
    mesh.invalidateCaches();
}

Expected<Mesh> triangulateCameraPoints( const PointCloud & cloud, const CameraPointsTriangulationSettings & settings, const ProgressCallback & cb )
{
    return triangulateCameraPoints( cloud.points, &cloud.validPoints, settings, cb );
}

Expected<Mesh> triangulateCameraPoints( PointCloud && cloud, const CameraPointsTriangulationSettings & settings, const ProgressCallback & cb )
{
    return triangulateCameraPoints( std::move( cloud.points ), &cloud.validPoints, settings, cb );
}

} //namespace MR
