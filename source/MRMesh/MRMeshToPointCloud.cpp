#include "MRMeshToPointCloud.h"
#include "MRMesh.h"
#include "MRMeshNormals.h"
#include "MRBitSetParallelFor.h"
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

Expected<PointCloud> meshToDensePointCloud( const Mesh& mesh, float radius, ProgressCallback cb )
{
    MR_TIMER;
    if ( !( radius > 0 ) )
        return unexpected( "meshToDensePointCloud: radius must be positive" );
    const float diameter = 2 * radius;
    const auto& topology = mesh.topology;

    // in how many equal parts each side of the triangle is divided to split it in a grid of similar triangles,
    // each covered by its own three corners; the number is a power of two, which makes the grid of a face
    // conforming with (possibly finer) divisions of the face's edges
    Vector<int, FaceId> faceDivs( topology.faceSize(), 0 );
    if ( !BitSetParallelFor( topology.getValidFaces(), [&]( FaceId f )
    {
        Vector3f v[3];
        mesh.getTriPoints( f, v );
        // no point of a triangle is farther from the nearest vertex than the radius of the minimal enclosing
        // circle, which is the circumcircle for non-obtuse triangles and half of the longest edge otherwise
        faceDivs[f] = ceilPow2( std::sqrt( mincircleDiameterSq( v[0], v[1], v[2] ) ) / diameter );
    }, subprogress( cb, 0.0f, 0.1f ) ) )
        return unexpectedOperationCanceled();

    // in how many equal parts each edge is divided: enough to make every part not longer than 2*radius,
    // and not less than the incident faces divide it
    Vector<int, UndirectedEdgeId> edgeDivs( topology.undirectedEdgeSize(), 0 );
    if ( !ParallelFor( 0_ue, edgeDivs.endId(), [&]( UndirectedEdgeId ue )
    {
        const EdgeId e = ue;
        if ( topology.isLoneEdge( e ) )
            return;
        int divs = ceilPow2( mesh.edgeLength( ue ) / diameter );
        for ( auto f : { topology.left( e ), topology.right( e ) } )
            if ( f )
                divs = std::max( divs, faceDivs[f] );
        edgeDivs[ue] = divs;
    }, subprogress( cb, 0.1f, 0.2f ) ) )
        return unexpectedOperationCanceled();

    // the samples of every edge and every face occupy a dedicated range in the resulting cloud
    size_t numPoints = mesh.points.size();
    Vector<size_t, UndirectedEdgeId> edgeOffset( edgeDivs.size() );
    for ( auto ue = 0_ue; ue < edgeOffset.endId(); ++ue )
    {
        edgeOffset[ue] = numPoints;
        if ( edgeDivs[ue] > 1 )
            numPoints += edgeDivs[ue] - 1;
    }
    Vector<size_t, FaceId> faceOffset( faceDivs.size() );
    for ( auto f = 0_f; f < faceOffset.endId(); ++f )
    {
        faceOffset[f] = numPoints;
        if ( const auto divs = faceDivs[f]; divs > 2 )
            numPoints += size_t( divs - 1 ) * ( divs - 2 ) / 2;
    }

    PointCloud res;
    res.points = mesh.points;
    res.points.resize( numPoints );
    res.validPoints = topology.getValidVerts();
    res.validPoints.resize( mesh.points.size(), false );
    res.validPoints.resize( numPoints, true );

    if ( !ParallelFor( 0_ue, edgeDivs.endId(), [&]( UndirectedEdgeId ue )
    {
        const auto divs = edgeDivs[ue];
        auto v = VertId( edgeOffset[ue] );
        for ( int i = 1; i < divs; ++i )
            res.points[v++] = mesh.edgePoint( EdgeId( ue ), float( i ) / divs );
    }, subprogress( cb, 0.2f, 0.6f ) ) )
        return unexpectedOperationCanceled();

    if ( !BitSetParallelFor( topology.getValidFaces(), [&]( FaceId f )
    {
        const auto divs = faceDivs[f];
        if ( divs <= 2 )
            return; // the grid of this face has no points strictly inside it
        Vector3f v[3];
        mesh.getTriPoints( f, v );
        const auto d1 = ( v[1] - v[0] ) / float( divs );
        const auto d2 = ( v[2] - v[0] ) / float( divs );
        auto p = VertId( faceOffset[f] );
        for ( int i = 1; i + 2 <= divs; ++i )
            for ( int j = 1; i + j + 1 <= divs; ++j )
                res.points[p++] = v[0] + float( i ) * d1 + float( j ) * d2;
    }, subprogress( cb, 0.6f, 1.0f ) ) )
        return unexpectedOperationCanceled();

    return res;
}

}
