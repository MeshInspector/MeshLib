#include "MRMeshToPointCloud.h"
#include "MRMesh.h"
#include "MRMeshNormals.h"
#include "MRBitSetParallelFor.h"
#include "MRBuffer.h"
#include "MRMeshPart.h"
#include "MRRegionBoundary.h"
#include "MRParallelFor.h"
#include "MRTriMath.h"
#include "MRTimer.h"

namespace MR
{

PointCloud meshToPointCloud( const Mesh& mesh, bool saveNormals /*= true */, const VertBitSet* verts )
{
    PointCloud res;
    res.points = mesh.points;
    res.validPoints = mesh.topology.getVertIds( verts );
    if(saveNormals)
        res.normals = computePerVertNormals( mesh );
    return res;
}

namespace
{

/// returns the smallest power of two, which is not less than given value (and not less than 1)
int ceilPow2( float v )
{
    int res = 1;
    while ( res < v && res < ( 1 << 24 ) )
        res <<= 1;
    return res;
}

} // anonymous namespace

Expected<PointCloud> meshToDensePointCloud( const MeshPart& mp, float radius, bool saveNormals, const ProgressCallback& cb )
{
    MR_TIMER;
    if ( !( radius > 0 ) )
        return unexpected( "meshToDensePointCloud: radius must be positive" );
    const float diameter = 2 * radius;
    const auto& mesh = mp.mesh;
    const auto& topology = mesh.topology;
    const auto& faces = topology.getFaceIds( mp.region );
    const auto [faceDivsCb, edgeDivsCb, edgePointsCb, facePointsCb] = splitProgress( cb, 0.1f, 0.2f, 0.6f );

    // in how many equal parts each side of the triangle is divided to split it in a grid of similar triangles,
    // each covered by its own three corners; the number is a power of two, which makes the grid of a face
    // conforming with (possibly finer) divisions of the face's edges
    Buffer<int, FaceId> faceDivs( topology.faceSize() );
    if ( !BitSetParallelFor( faces, [&]( FaceId f )
    {
        Vector3f v[3];
        mesh.getTriPoints( f, v );
        // no point of a triangle is farther from the nearest vertex than the radius of the minimal enclosing
        // circle, which is the circumcircle for non-obtuse triangles and half of the longest edge otherwise
        faceDivs[f] = ceilPow2( std::sqrt( mincircleDiameterSq( v[0], v[1], v[2] ) ) / diameter );
    }, faceDivsCb ) )
        return unexpectedOperationCanceled();

    // in how many equal parts each edge is divided: enough to make every part not longer than 2*radius,
    // and not less than the sampled incident faces divide it; the edges without such faces
    // (including the lone ones) are not sampled at all
    Buffer<int, UndirectedEdgeId> edgeDivs( topology.undirectedEdgeSize() );
    if ( !ParallelFor( 0_ue, edgeDivs.endId(), [&]( UndirectedEdgeId ue )
    {
        const EdgeId e = ue;
        int divs = 0;
        for ( auto f : { topology.left( e ), topology.right( e ) } )
            if ( f && faces.test( f ) )
                divs = std::max( divs, faceDivs[f] );
        if ( divs > 0 )
            divs = std::max( divs, ceilPow2( mesh.edgeLength( ue ) / diameter ) );
        edgeDivs[ue] = divs;
    }, edgeDivsCb ) )
        return unexpectedOperationCanceled();

    // the samples of every edge and every face occupy a dedicated range in the resulting cloud;
    // the arrays here are not initialized, and the elements of missing faces are never touched at all
    size_t numPoints = mesh.points.size();
    Buffer<size_t, UndirectedEdgeId> edgeOffset( edgeDivs.size() );
    for ( auto ue = 0_ue; ue < edgeDivs.endId(); ++ue )
    {
        edgeOffset[ue] = numPoints;
        if ( edgeDivs[ue] > 1 )
            numPoints += edgeDivs[ue] - 1;
    }
    Buffer<size_t, FaceId> faceOffset( faceDivs.size() );
    for ( auto f : faces )
    {
        faceOffset[f] = numPoints;
        if ( const auto divs = faceDivs[f]; divs > 2 )
            numPoints += size_t( divs - 1 ) * ( divs - 2 ) / 2;
    }

    PointCloud res;
    res.points = mesh.points;
    res.points.resizeNoInit( numPoints ); // the samples of the edges and the faces are set below
    VertBitSet store;
    res.validPoints = getIncidentVerts( topology, mp.region, store );
    res.validPoints.resize( mesh.points.size(), false );
    res.validPoints.resize( numPoints, true );
    if ( saveNormals )
    {
        res.normals = computePerVertNormals( mesh );
        res.normals.resizeNoInit( numPoints );
    }

    if ( !ParallelFor( 0_ue, edgeDivs.endId(), [&]( UndirectedEdgeId ue )
    {
        const auto divs = edgeDivs[ue];
        if ( divs <= 1 )
            return;
        const EdgeId e = ue;
        Vector3f nOrg, nDest;
        if ( saveNormals )
        {
            nOrg = res.normals[ topology.org( e ) ];
            nDest = res.normals[ topology.dest( e ) ];
        }
        auto v = VertId( edgeOffset[ue] );
        for ( int i = 1; i < divs; ++i, ++v )
        {
            const float t = float( i ) / divs;
            res.points[v] = mesh.edgePoint( e, t );
            if ( saveNormals )
                res.normals[v] = ( ( 1 - t ) * nOrg + t * nDest ).normalized();
        }
    }, edgePointsCb ) )
        return unexpectedOperationCanceled();

    if ( !BitSetParallelFor( faces, [&]( FaceId f )
    {
        const auto divs = faceDivs[f];
        if ( divs <= 2 )
            return; // the grid of this face has no points strictly inside it
        Vector3f v[3];
        mesh.getTriPoints( f, v );
        Vector3f n[3];
        if ( saveNormals )
        {
            const auto vs = topology.getTriVerts( f );
            for ( int i = 0; i < 3; ++i )
                n[i] = res.normals[ vs[i] ];
        }
        auto p = VertId( faceOffset[f] );
        for ( int i = 1; i + 2 <= divs; ++i )
            for ( int j = 1; i + j + 1 <= divs; ++j, ++p )
            {
                const float a = float( i ) / divs, b = float( j ) / divs;
                res.points[p] = v[0] + a * ( v[1] - v[0] ) + b * ( v[2] - v[0] );
                if ( saveNormals )
                    res.normals[p] = ( ( 1 - a - b ) * n[0] + a * n[1] + b * n[2] ).normalized();
            }
    }, facePointsCb ) )
        return unexpectedOperationCanceled();

    return res;
}

}
