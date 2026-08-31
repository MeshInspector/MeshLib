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

/// returns the smallest power of two n (and not less than 1) with n * n * bSq >= aSq, that is
/// n >= sqrt( aSq / bSq ): no square root and no division are needed, and quadrupling is exact
int ceilPow2Sq( float aSq, float bSq )
{
    int res = 1;
    while ( bSq < aSq && res < ( 1 << 24 ) )
    {
        res <<= 1;
        bSq *= 4;
    }
    return res;
}

/// returns the smallest n (and not less than 1) with n * n * bSq >= aSq, that is n >= sqrt( aSq / bSq )
int ceilSq( float aSq, float bSq )
{
    int res = 1;
    while ( float( res ) * float( res ) * bSq < aSq && res < ( 1 << 24 ) )
        ++res;
    return res;
}

} // anonymous namespace

Expected<PointCloud> meshToDensePointCloud( const MeshPart& mp, float radius, bool saveNormals, const ProgressCallback& cb )
{
    MR_TIMER;
    if ( !( radius > 0 ) )
        return unexpected( "meshToDensePointCloud: radius must be positive" );
    const float radiusSq = radius * radius;
    const auto& mesh = mp.mesh;
    const auto& topology = mesh.topology;
    const auto& faces = topology.getFaceIds( mp.region );
    // whole mesh: left/right of an edge is never a face outside faces, so it needs no test
    const bool wholeMesh = !mp.region;
    const auto [faceDivsCb, edgeDivsCb, edgePointsCb, facePointsCb] = splitProgress( cb, 0.1f, 0.2f, 0.6f );

    // in how many equal parts each side of the triangle is divided to split it in a grid of similar
    // triangles, each covered by its own three corners; the number is a power of two, which makes the
    // grid of a face conforming with (possibly finer) divisions of the face's edges
    Buffer<int, FaceId> faceDivs( topology.faceSize() );
    // a thin triangle is covered by the samples of its longest edge alone instead: nothing is put
    // inside it, and only that edge, kept here with the division it needs, has to be divided
    Buffer<int, FaceId> faceStripDivs( topology.faceSize() );
    Buffer<UndirectedEdgeId, FaceId> faceStripEdge( topology.faceSize() );
    if ( !BitSetParallelFor( faces, [&]( FaceId f )
    {
        faceStripEdge[f] = UndirectedEdgeId{};
        Vector3f v[3];
        mesh.getTriPoints( f, v );
        // by definition no point of a triangle is farther from the nearest vertex than the covering
        // radius, so a triangle with a smaller one is covered by its own vertices and is not divided;
        // the minimal enclosing circle bounds that radius from above and is cheaper to find
        if ( mincircleDiameterSq( v[0], v[1], v[2] ) <= 4 * radiusSq )
        {
            faceDivs[f] = 1;
            return;
        }
        const int divs = ceilPow2Sq( coveringRadiusSq( v[0], v[1], v[2] ), radiusSq );
        faceDivs[f] = divs;

        // every point of a triangle projects on its longest edge inside it, because the angles at the
        // ends of that edge are acute; such a point is h away from the edge and no farther than half
        // a division step along it, so the samples of that edge alone cover the triangle if h < radius
        EdgeId es[3];
        topology.getTriEdges( f, es );
        int longest = 0;
        auto longestSq = ( v[1] - v[0] ).lengthSq();
        for ( int i = 1; i < 3; ++i )
        {
            const auto lenSq = ( v[( i + 1 ) % 3] - v[i] ).lengthSq();
            if ( lenSq > longestSq )
            {
                longest = i;
                longestSq = lenSq;
            }
        }
        // twice the area over the length of that edge, squared
        const auto hSq = cross( v[1] - v[0], v[2] - v[0] ).lengthSq() / longestSq;
        if ( hSq >= radiusSq )
            return;
        // unlike the grid, the strip needs no conforming nodes on the edge, only a small enough
        // step along it, so any number of parts will do and this one is not rounded to a power of two
        const int stripDivs = ceilSq( longestSq, 4 * ( radiusSq - hSq ) );
        // the grid costs the samples of its three edges and of its interior, the strip only its edge
        if ( stripDivs - 1 >= 3 * ( divs - 1 ) + ( divs - 1 ) * ( divs - 2 ) / 2 )
            return;
        faceDivs[f] = 1;
        faceStripDivs[f] = stripDivs;
        faceStripEdge[f] = es[longest].undirected();
    }, faceDivsCb ) )
        return unexpectedOperationCanceled();

    // in how many equal parts each edge is divided: as much as the sampled faces incident to it
    // divide it, and not at all if there are no such faces (including the lone edges); the edges
    // of an undivided face need no samples of their own, because that face covers them already
    Buffer<int, UndirectedEdgeId> edgeDivs( topology.undirectedEdgeSize() );
    if ( !ParallelFor( 0_ue, edgeDivs.endId(), [&]( UndirectedEdgeId ue )
    {
        const EdgeId e = ue;
        int grid = 0, strip = 0;
        for ( auto f : { topology.left( e ), topology.right( e ) } )
        {
            if ( !f || !( wholeMesh || faces.test( f ) ) )
                continue;
            grid = std::max( grid, faceDivs[f] );
            if ( faceStripEdge[f] == ue )
                strip = std::max( strip, faceStripDivs[f] );
        }
        // the nodes of a face's grid on this edge must be among its division points, so the number of
        // parts stays a multiple of what the grids ask, and grows to what the strips ask
        edgeDivs[ue] = strip <= grid ? grid
            : grid <= 1 ? strip : ( ( strip + grid - 1 ) / grid ) * grid;
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
    res.validPoints = wholeMesh ? topology.getValidVerts() : getIncidentVerts( topology, faces );
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
