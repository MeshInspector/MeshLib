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

/// how a face is sampled: by a grid of triangles similar to it, or in rows parallel to its longest
/// edge, or by the samples of that edge alone, whichever of the three costs the fewest samples
struct FaceLayout
{
    int base;   ///< local index of the longest edge, going from v[base] to v[(base+1)%3]
    int grid;   ///< in how many parts every side is divided for a grid; zero if it is not a grid
    int wants;  ///< in how many parts the face wants its longest edge; zero if it does not care
    bool rows;  ///< the face is sampled in rows parallel to its longest edge

    /// the three vertices alone cover such a face, and it needs no samples of any kind
    [[nodiscard]] bool covered() const { return grid <= 0 && wants <= 0 && !rows; }
};

/// the samples within a row are this far from each other, on every face
float rowSampleStep( float radius )
{
    return radius * cSqrt2;
}

/// the rows parallel to the longest edge of a face, in the fractions of the height over that edge
struct RowLayout
{
    int num = 0;     ///< the number of rows
    float first = 0; ///< where the first one is
    float step = 0;  ///< the distance between them
};

/// lays out the rows over a longest edge of the given length divided in the given number of parts,
/// with the opposite vertex at the given height above it
RowLayout layoutRows( float baseLen, float height, float radius, float radiusSq, int baseDivs )
{
    RowLayout res;
    // the samples of the base are not farther than half a step from any point of it, so they reach
    // sqrt( radius^2 - (step/2)^2 ) up in the height, and a face flatter than that needs no rows
    const float baseStep = baseLen / baseDivs;
    const float first = std::sqrt( std::max( 0.0f, radiusSq - 0.25f * baseStep * baseStep ) );
    if ( height <= first )
        return res;
    // a point above the rows is within a band from the row below it, which is never narrower than
    // the face is there, and within half a step along that row; sqrt(2)*radius along the row
    // against radius/sqrt(2) between the rows keeps both within the radius with the fewest samples
    res.num = divsForStep( height - first, radius / cSqrt2 );
    res.first = first / height;
    res.step = ( height - first ) / ( res.num * height );
    return res;
}

/// the number of samples inside a face sampled in the given rows
int numRowSamples( const RowLayout & l, float baseLen, float radius )
{
    int res = 0;
    for ( int i = 0; i < l.num; ++i )
    {
        const float hf = l.first + i * l.step;
        res += divsForStep( baseLen * ( 1 - hf ), rowSampleStep( radius ) ) + 1;
    }
    return res;
}

/// chooses how a face is sampled, and what it needs of its longest edge
FaceLayout layoutFace( const Vector3f v[3], float radius, float radiusSq )
{
    FaceLayout res{ 0, 0, 0, false };
    // no point of a triangle is farther from the nearest vertex than the covering radius, and the
    // minimal enclosing circle bounds that radius from above and is cheaper to find
    if ( mincircleDiameterSq( v[0], v[1], v[2] ) <= 4 * radiusSq )
        return res;
    const auto coverSq = coveringRadiusSq( v[0], v[1], v[2] );
    if ( coverSq <= radiusSq )
        return res;

    // the longest edge is the base: the angles at its ends are acute, so every point of the face
    // projects on it inside it, and the sections parallel to it shrink towards the opposite vertex,
    // which makes a row below a point never narrower than the face is at that point
    float len[3];
    int selfDivs[3];
    for ( int i = 0; i < 3; ++i )
        len[i] = ( v[( i + 1 ) % 3] - v[i] ).length();
    res.base = ( len[0] >= len[1] && len[0] >= len[2] ) ? 0 : ( len[1] >= len[2] ? 1 : 2 );
    const float baseLen = len[res.base];
    const float height = cross( v[1] - v[0], v[2] - v[0] ).length() / baseLen; // 2*area over the base
    for ( int i = 0; i < 3; ++i )
        selfDivs[i] = divsForStep( len[i], 2 * radius ); // enough for an edge to cover itself

    // a grid of triangles similar to the face is covered by its own corners, but its nodes on an
    // edge must be among the division points of that edge, which is why the number is rounded to a
    // power of two: the maximum of two such numbers is a multiple of both
    // the three costs below count the samples inside the face plus half of those on its edges,
    // since an edge is shared, and a grid needs no self-covering edges: it covers them itself
    const int grid = ceilPow2Sq( coverSq, radiusSq );
    int gridCost = 2 * ( grid - 1 ) * ( grid - 2 ) / 2 + 3 * ( grid - 1 );

    // the samples of the base alone cover a face flatter than the radius, once it is divided finely
    int stripCost = std::numeric_limits<int>::max(), wants = 0;
    if ( height < radius )
    {
        wants = divsForStep( baseLen, 2 * std::sqrt( radiusSq - height * height ) );
        stripCost = wants - 1;
    }

    const auto rowLayout = layoutRows( baseLen, height, radius, radiusSq, selfDivs[res.base] );
    int rowsCost = 2 * numRowSamples( rowLayout, baseLen, radius );
    for ( int i = 0; i < 3; ++i )
        rowsCost += selfDivs[i] - 1;

    if ( stripCost <= rowsCost && stripCost <= gridCost )
        res.wants = wants;
    else if ( gridCost < rowsCost )
        res.grid = grid;
    else if ( rowLayout.num > 0 )
        res.rows = true;
    else
        // the rows are the cheapest and there are none of them: the samples of the longest edge
        // already cover the face, but it has to ask for them, or it will look covered by its
        // vertices and that edge will not be divided at all
        res.wants = divsForStep( baseLen, 2 * radius );
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
    const auto [layoutCb, edgeDivsCb, samplesCb, edgePointsCb, facePointsCb] =
        splitProgress( cb, 0.1f, 0.2f, 0.25f, 0.6f );

    Buffer<FaceLayout, FaceId> layouts( topology.faceSize() );
    if ( !BitSetParallelFor( faces, [&]( FaceId f )
    {
        Vector3f v[3];
        mesh.getTriPoints( f, v );
        layouts[f] = layoutFace( v, radius, radiusSq );
    }, layoutCb ) )
        return unexpectedOperationCanceled();

    // an edge is divided only if some incident face relies on that: a face covered by its own
    // vertices does not, and a face covered by its longest edge relies on that edge alone. The
    // number of parts is a multiple of what the grids ask, since their nodes must be among the
    // division points, and not less than what the others ask, which needs no divisibility
    Buffer<int, UndirectedEdgeId> edgeDivs( topology.undirectedEdgeSize() );
    if ( !ParallelFor( 0_ue, edgeDivs.endId(), [&]( UndirectedEdgeId ue )
    {
        const EdgeId e = ue;
        int gridAsk = 0, plainAsk = 0;
        for ( auto f : { topology.left( e ), topology.right( e ) } )
        {
            if ( !f || !( wholeMesh || faces.test( f ) ) )
                continue;
            const auto & l = layouts[f];
            if ( l.covered() )
                continue;
            EdgeId es[3];
            topology.getTriEdges( f, es );
            const bool isBase = es[l.base].undirected() == ue;
            if ( l.grid > 0 )
                gridAsk = std::max( gridAsk, l.grid );
            else if ( l.rows || isBase ) // the rows rely on all three edges covering themselves
                plainAsk = std::max( plainAsk, divsForStep( mesh.edgeLength( ue ), 2 * radius ) );
            if ( isBase )
                plainAsk = std::max( plainAsk, l.wants );
        }
        if ( gridAsk <= 0 )
            edgeDivs[ue] = plainAsk;
        else // the smallest multiple of what the grids ask that is not less than the rest
            edgeDivs[ue] = gridAsk * ( ( std::max( plainAsk, gridAsk ) + gridAsk - 1 ) / gridAsk );
    }, edgeDivsCb ) )
        return unexpectedOperationCanceled();

    // the rows of a face are laid out over its longest edge as divided above, which may leave it
    // with fewer of them than the face expected, or with none at all
    auto rowsOf = [&]( FaceId f, const Vector3f v[3], float & baseLen )
    {
        const auto & l = layouts[f];
        EdgeId es[3];
        topology.getTriEdges( f, es );
        baseLen = ( v[( l.base + 1 ) % 3] - v[l.base] ).length();
        const float height = cross( v[1] - v[0], v[2] - v[0] ).length() / baseLen;
        return layoutRows( baseLen, height, radius, radiusSq, edgeDivs[es[l.base].undirected()] );
    };

    Buffer<int, FaceId> faceSamples( topology.faceSize() );
    if ( !BitSetParallelFor( faces, [&]( FaceId f )
    {
        const auto & l = layouts[f];
        if ( l.grid > 0 )
        {
            faceSamples[f] = ( l.grid - 1 ) * ( l.grid - 2 ) / 2;
            return;
        }
        faceSamples[f] = 0;
        if ( !l.rows )
            return; // nothing inside: the vertices or the longest edge alone cover this face
        Vector3f v[3];
        mesh.getTriPoints( f, v );
        float baseLen = 0;
        const auto rows = rowsOf( f, v, baseLen ); // sets baseLen, so not in the call below
        faceSamples[f] = numRowSamples( rows, baseLen, radius );
    }, samplesCb ) )
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
        if ( faceSamples[f] <= 0 )
            return;
        const auto & l = layouts[f];
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
        if ( l.grid > 0 )
        {
            const int divs = l.grid;
            for ( int i = 1; i + 2 <= divs; ++i )
                for ( int j = 1; i + j + 1 <= divs; ++j, ++p )
                {
                    const float a = float( i ) / divs, b = float( j ) / divs;
                    res.points[p] = v[0] + a * ( v[1] - v[0] ) + b * ( v[2] - v[0] );
                    if ( saveNormals )
                        res.normals[p] = ( ( 1 - a - b ) * n[0] + a * n[1] + b * n[2] ).normalized();
                }
            return;
        }
        const int bi = l.base, bj = ( l.base + 1 ) % 3, bk = ( l.base + 2 ) % 3;
        float baseLen = 0;
        const auto rows = rowsOf( f, v, baseLen );
        for ( int i = 0; i < rows.num; ++i )
        {
            const float hf = rows.first + i * rows.step;
            const auto rowOrg = v[bi] + hf * ( v[bk] - v[bi] );
            const auto rowDest = v[bj] + hf * ( v[bk] - v[bj] );
            const int divs = divsForStep( baseLen * ( 1 - hf ), rowSampleStep( radius ) );
            for ( int j = 0; j <= divs; ++j, ++p )
            {
                const float g = float( j ) / divs;
                res.points[p] = rowOrg + g * ( rowDest - rowOrg );
                if ( saveNormals )
                    res.normals[p] = ( ( 1 - hf ) * ( 1 - g ) * n[bi]
                        + ( 1 - hf ) * g * n[bj] + hf * n[bk] ).normalized();
            }
        }
    }, facePointsCb ) )
        return unexpectedOperationCanceled();

    return res;
}

}
