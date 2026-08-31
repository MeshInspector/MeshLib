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

constexpr float cSqrt2 = 1.41421356f;

/// returns the smallest number of equal parts, and not less than one, in which a segment of the
/// given length has to be divided to make every part not longer than the given step
int divsForStep( float len, float step )
{
    if ( !( len > 0 ) || !( step > 0 ) )
        return 1;
    int res = int( len / step );
    while ( res * step < len && res < ( 1 << 24 ) )
        ++res;
    return std::max( 1, res );
}

/// how a face is sampled inside: in rows parallel to its longest edge
struct FaceLayout
{
    int base;      ///< local index of the longest edge, going from v[base] to v[(base+1)%3]
    int rows;      ///< the number of rows; zero if the longest edge alone covers the face
    float first;   ///< the height of the first row over that edge
    float band;    ///< the distance between the rows
    float height;  ///< the height of the opposite vertex over that edge
    float step;    ///< the distance between the samples within a row
    bool covered;  ///< the three vertices alone cover the face, so it needs no samples at all
};

FaceLayout layoutFace( const Vector3f v[3], float radius, float radiusSq )
{
    FaceLayout res{ 0, 0, 0, 0, 0, 0, false };
    // no point of a triangle is farther from the nearest vertex than the covering radius, and the
    // minimal enclosing circle bounds that radius from above and is cheaper to find
    if ( mincircleDiameterSq( v[0], v[1], v[2] ) <= 4 * radiusSq
        || coveringRadiusSq( v[0], v[1], v[2] ) <= radiusSq )
    {
        res.covered = true;
        return res;
    }

    // the longest edge is the base: the angles at its ends are acute, so every point of the face
    // projects on it inside it, and the sections parallel to it shrink towards the opposite vertex,
    // which makes a row below a point never narrower than the face is at that point
    float baseLenSq = 0;
    for ( int i = 0; i < 3; ++i )
    {
        const auto lenSq = ( v[( i + 1 ) % 3] - v[i] ).lengthSq();
        if ( lenSq > baseLenSq )
        {
            baseLenSq = lenSq;
            res.base = i;
        }
    }
    const float baseLen = std::sqrt( baseLenSq );
    res.height = cross( v[1] - v[0], v[2] - v[0] ).length() / baseLen; // twice the area over the base

    // no point of the base is farther than half a step from a sample of it, so those samples reach
    // this far in the height; a face lower than that is covered by its longest edge alone
    const float baseStep = baseLen / divsForStep( baseLen, 2 * radius );
    res.first = std::sqrt( std::max( 0.0f, radiusSq - 0.25f * baseStep * baseStep ) );
    if ( res.height <= res.first )
        return res;

    // a point above the rows is within a band from the row below it and within half a step along
    // that row; sqrt(2)*radius along the row against radius/sqrt(2) between the rows keeps both
    // within the radius and gives the fewest samples
    res.step = radius * cSqrt2;
    res.rows = divsForStep( res.height - res.first, radius / cSqrt2 );
    res.band = ( res.height - res.first ) / res.rows;
    return res;
}

/// the number of samples inside a face with the given layout
int numFaceSamples( const FaceLayout & l, float baseLen )
{
    int res = 0;
    for ( int i = 0; i < l.rows; ++i )
    {
        const float hf = ( l.first + i * l.band ) / l.height;
        res += divsForStep( baseLen * ( 1 - hf ), l.step ) + 1;
    }
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
    const auto [faceLayoutCb, edgeDivsCb, edgePointsCb, facePointsCb] = splitProgress( cb, 0.1f, 0.2f, 0.6f );

    Buffer<FaceLayout, FaceId> layouts( topology.faceSize() );
    Buffer<int, FaceId> faceSamples( topology.faceSize() );
    if ( !BitSetParallelFor( faces, [&]( FaceId f )
    {
        Vector3f v[3];
        mesh.getTriPoints( f, v );
        const auto l = layoutFace( v, radius, radiusSq );
        layouts[f] = l;
        faceSamples[f] = l.rows <= 0 ? 0
            : numFaceSamples( l, ( v[( l.base + 1 ) % 3] - v[l.base] ).length() );
    }, faceLayoutCb ) )
        return unexpectedOperationCanceled();

    // an edge is divided so that every point of it is within the radius from a sample of its own,
    // and only if some incident face relies on that: a face covered by its vertices does not, and a
    // face covered by its longest edge alone relies on that edge only
    Buffer<int, UndirectedEdgeId> edgeDivs( topology.undirectedEdgeSize() );
    if ( !ParallelFor( 0_ue, edgeDivs.endId(), [&]( UndirectedEdgeId ue )
    {
        const EdgeId e = ue;
        bool need = false;
        for ( auto f : { topology.left( e ), topology.right( e ) } )
        {
            if ( !f || !( wholeMesh || faces.test( f ) ) )
                continue;
            const auto & l = layouts[f];
            if ( l.covered )
                continue;
            if ( l.rows > 0 )
                need = true;
            else
            {
                EdgeId es[3];
                topology.getTriEdges( f, es );
                if ( es[l.base].undirected() == ue )
                    need = true;
            }
        }
        edgeDivs[ue] = need ? divsForStep( mesh.edgeLength( ue ), 2 * radius ) : 0;
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
    Buffer<size_t, FaceId> faceOffset( layouts.size() );
    for ( auto f : faces )
    {
        faceOffset[f] = numPoints;
        numPoints += faceSamples[f];
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
        const auto & l = layouts[f];
        if ( l.rows <= 0 )
            return;
        Vector3f v[3];
        mesh.getTriPoints( f, v );
        const auto & a = v[l.base], & b = v[( l.base + 1 ) % 3], & c = v[( l.base + 2 ) % 3];
        Vector3f na, nb, nc;
        if ( saveNormals )
        {
            const auto vs = topology.getTriVerts( f );
            na = res.normals[ vs[l.base] ];
            nb = res.normals[ vs[( l.base + 1 ) % 3] ];
            nc = res.normals[ vs[( l.base + 2 ) % 3] ];
        }
        const float baseLen = ( b - a ).length();
        auto p = VertId( faceOffset[f] );
        for ( int i = 0; i < l.rows; ++i )
        {
            const float hf = ( l.first + i * l.band ) / l.height;
            const auto rowOrg = a + hf * ( c - a );
            const auto rowDest = b + hf * ( c - b );
            const int divs = divsForStep( baseLen * ( 1 - hf ), l.step );
            for ( int j = 0; j <= divs; ++j, ++p )
            {
                const float g = float( j ) / divs;
                res.points[p] = rowOrg + g * ( rowDest - rowOrg );
                if ( saveNormals )
                    res.normals[p] = ( ( 1 - hf ) * ( 1 - g ) * na
                        + ( 1 - hf ) * g * nb + hf * nc ).normalized();
            }
        }
    }, facePointsCb ) )
        return unexpectedOperationCanceled();

    return res;
}

}
