#include "MRRegularGridMesh.h"
#include "MRMeshBuilder.h"
#include "MRTriMath.h"
#include "MRVector2.h"
#include "MRMeshDelone.h"
#include "MRGridSettings.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRGTest.h"

namespace MR
{

Expected<Mesh> makeRegularGridMesh( size_t width, size_t height,
                          const RegularGridLatticeValidator& validator, 
                          const RegularGridLatticePositioner& positioner,
                          const RegularGridMeshFaceValidator& faceValidator,
                          ProgressCallback cb )
{
    MR_TIMER
    Mesh res;

    GridSettings gs =
    {
        .dim = Vector2i( (int)width - 1, (int)height - 1 )
    };

    BitSet validGridVerts( width * height );
    gs.vertIds.b.resize( width * height );
    auto result = BitSetParallelForAll( validGridVerts, [&]( size_t p )
    {
        auto y = p / width;
        auto x = p - y * width;
        if ( validator( x, y ) )
            validGridVerts.set( p );
        else
            gs.vertIds.b[p] = VertId{};
    }, subprogress( cb, 0.0f, 0.1f ) );

    if ( !result )
        return unexpectedOperationCanceled();

    VertId nextVertId{ 0 };
    for ( auto p : validGridVerts )
        gs.vertIds.b[p] = nextVertId++;

    const auto vertSize = size_t( nextVertId );
    gs.vertIds.tsize = vertSize;
    res.points.resize( vertSize );
    result = BitSetParallelFor( validGridVerts, [&]( size_t p )
    {
        auto y = p / width;
        auto x = p - y * width;
        res.points[gs.vertIds.b[p]] = positioner( x, y );
    }, subprogress( cb, 0.1f, 0.2f ) );

    if ( !result )
        return unexpectedOperationCanceled();

    BitSet validLoUpTris( 2 * ( width - 1 ) * ( height - 1 ) );
    BitSet diagonalA( ( width - 1 ) * ( height - 1 ) ); // if one of triangles is valid

    auto getVertId = [&]( Vector2i pos ) -> VertId
    {
        if ( pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height )
            return VertId();
        return gs.vertIds.b[pos.x + pos.y * width];
    };

    gs.faceIds.b.resize( validLoUpTris.size() );
    result = BitSetParallelForAll( diagonalA, [&]( size_t p )
    {
        const auto y = int( p / ( width - 1 ) );
        const auto x = int( p - y * ( width - 1 ) );
        const Vector2i p00{ x, y };
        const Vector2i p01{ x, y + 1 };
        const Vector2i p10{ x + 1, y };
        const Vector2i p11{ x + 1, y + 1 };

        const auto v00 = getVertId( p00 );
        const auto v01 = getVertId( p01 );
        const auto v10 = getVertId( p10 );
        const auto v11 = getVertId( p11 );
        const auto count = v00.valid() + v01.valid() + v10.valid() + v11.valid();

        auto createFace = [&]( Vector2i a, Vector2i b, Vector2i c, size_t fidx )
        {
            if ( !faceValidator || faceValidator( a.x, a.y, b.x, b.y, c.x, c.y ) )
                validLoUpTris.set( fidx );
            else
                gs.faceIds.b[fidx] = FaceId{};
        };

        switch ( count )
        {
        case 4:
            // two possible triangles
            if ( checkDeloneQuadrangle( res.points[v00], res.points[v01], res.points[v11], res.points[v10] ) )
            {
                diagonalA.set( p );
                createFace( p11, p01, p00, 2 * p + 1 ); //upper
                createFace( p11, p00, p10, 2 * p ); //lower
            }
            else
            {
                createFace( p01, p00, p10, 2 * p ); //lower
                createFace( p01, p10, p11, 2 * p + 1 ); //upper
            }
            break;
        case 3:
            // one possible triangle
            if ( !v00 )
            {
                createFace( p01, p10, p11, 2 * p + 1 ); //upper
                gs.faceIds.b[2 * p] = FaceId{};
            }
            else if ( !v01 )
            {
                diagonalA.set( p );
                createFace( p11, p00, p10, 2 * p ); //lower
                gs.faceIds.b[2 * p + 1] = FaceId{};
            }
            else if ( !v10 )
            {
                diagonalA.set( p );
                createFace( p11, p01, p00, 2 * p + 1 ); //upper
                gs.faceIds.b[2 * p] = FaceId{};
            }
            else if ( !v11 )
            {
                createFace( p01, p00, p10, 2 * p ); //lower
                gs.faceIds.b[2 * p + 1] = FaceId{};
            }
            break;
        case 2:
        case 1:
        case 0:
            // no possible triangles
            gs.faceIds.b[2 * p] = FaceId{};
            gs.faceIds.b[2 * p + 1] = FaceId{};
            break;
        }
    }, subprogress( cb, 0.2f, 0.3f ) );

    if ( !result )
        return unexpectedOperationCanceled();

    FaceId nextFaceId{ 0 };
    for ( auto p : validLoUpTris )
        gs.faceIds.b[p] = nextFaceId++;
    gs.faceIds.tsize = size_t( nextFaceId );

    // return true if this edge has either left or right face
    auto hasEdge = [&]( Vector2i v, GridSettings::EdgeType et )
    {
        const auto hfidx = ( width - 1 ) * v.y + v.x;
        if ( et == GridSettings::EdgeType::Horizontal )
        {
            if ( v.x + 1 >= width )
                return false;
            return 
                ( v.y + 1 < height && validLoUpTris.test( 2 * hfidx ) ) ||
                ( v.y > 0 && validLoUpTris.test( 2 * ( hfidx - width + 1 ) + 1 ) );
        }

        if ( et == GridSettings::EdgeType::Vertical )
        {
            if ( v.y + 1 >= height )
                return false;
            if ( v.x + 1 < width )
            {
                if ( diagonalA.test( hfidx ) )
                {
                    if ( validLoUpTris.test( 2 * hfidx + 1 ) )
                        return true;
                }
                else
                {
                    if ( validLoUpTris.test( 2 * hfidx ) )
                        return true;
                }
            }
            if ( v.x > 0 )
            {
                if ( diagonalA.test( hfidx - 1 ) )
                {
                    if ( validLoUpTris.test( 2 * hfidx - 2 ) )
                        return true;
                }
                else
                {
                    if ( validLoUpTris.test( 2 * hfidx - 1 ) )
                        return true;
                }
            }
            return false;
        }
        assert( et == GridSettings::EdgeType::DiagonalA || et == GridSettings::EdgeType::DiagonalB );
        if ( v.x + 1 >= width || v.y + 1 >= height )
            return false;
        if ( !validLoUpTris.test( 2 * hfidx ) && !validLoUpTris.test( 2 * hfidx + 1 ) )
            return false;
        return diagonalA.test( hfidx ) == ( et == GridSettings::EdgeType::DiagonalA );
    };

    BitSet validGridEdges( 4 * width * height );
    gs.uedgeIds.b.resize( 4 * width * height );
    result = BitSetParallelForAll( validGridEdges, [&]( size_t loc )
    {
        auto p = loc / 4;
        auto et = GridSettings::EdgeType( loc - p * 4 );
        auto y = int( p / width );
        auto x = int( p - y * width );
        if ( hasEdge( { x, y }, et ) )
            validGridEdges.set( loc );
        else
            gs.uedgeIds.b[loc] = UndirectedEdgeId{};
    }, subprogress( cb, 0.3f, 0.4f ) );

    if ( !result )
        return unexpectedOperationCanceled();

    UndirectedEdgeId nextUEdgeId{ 0 };
    for ( auto p : validGridEdges )
        gs.uedgeIds.b[p] = nextUEdgeId++;
    gs.uedgeIds.tsize = size_t( nextUEdgeId );

    result = res.topology.buildGridMesh( gs, subprogress( cb, 0.4f, 0.8f ) );
    if ( !result )
        return unexpectedOperationCanceled();

    result = res.topology.checkValidity( subprogress( cb, 0.8f, 1.0f ) );
    if ( !result )
        return unexpectedOperationCanceled();

    return res;
}

Expected<Mesh> makeRegularGridMesh( VertCoords points, ProgressCallback cb )
{
    MR_TIMER
    tbb::parallel_sort( points.vec_.begin(), points.vec_.end(), [] ( const auto& l, const auto& r )
    {
        return l.y < r.y;
    } );
    if ( cb && !cb( 0.2f ) )
        return unexpectedOperationCanceled();

    std::vector<size_t> lineWidths;
    std::vector<size_t> positionOffsets;
    size_t gCounter = 0;
    size_t maxCounter = 0;
    while ( gCounter != points.size() )
    {
        size_t counter = 0;
        while ( points.vec_[gCounter + counter].y == points.vec_[gCounter].y )
            ++counter;
        positionOffsets.push_back( gCounter );
        lineWidths.push_back( counter );
        gCounter += counter;
        if ( counter > maxCounter )
            maxCounter = counter;
    }
    positionOffsets.push_back( gCounter );

    auto keepGoing = ParallelFor( positionOffsets, [&] ( size_t i )
    {
        if ( i + 1 == positionOffsets.size() )
            return;
        std::sort( points.vec_.begin() + positionOffsets[i], points.vec_.begin() + positionOffsets[i + 1],
            [] ( const auto& l, const auto& r ) { return l.x < r.x; } );
    }, subprogress( cb, 0.2f, 0.8f ) );
    
    if ( !keepGoing )
        return unexpectedOperationCanceled();

    auto res = makeRegularGridMesh( maxCounter, lineWidths.size(),
        [&] ( size_t x, size_t y )->bool
    {
        if ( y + 1 > lineWidths.size() )
            return false;
        if ( x + 1 > lineWidths[y] )
            return false;
        return true;
    },
        [&] ( size_t x, size_t y )->Vector3f
    {
        return points.vec_[positionOffsets[y] + x];
    }, {}, cb );

    if ( cb && !cb( 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

TEST(MRMesh, makeRegularGridMesh)
{
     auto m = makeRegularGridMesh( 2, 2,
         []( size_t, size_t ) { return true; },
         []( size_t x, size_t y ) { return Vector3f( (float)x, (float)y, 0 ); } ).value();
     ASSERT_TRUE( m.topology.checkValidity() );

     m = makeRegularGridMesh( 2, 3,
         []( size_t, size_t ) { return true; },
         []( size_t x, size_t y ) { return Vector3f( (float)x, (float)y, 0 ); } ).value();
     ASSERT_TRUE( m.topology.checkValidity() );

     m = makeRegularGridMesh( 5, 3,
         []( size_t, size_t ) { return true; },
         []( size_t x, size_t y ) { return Vector3f( (float)x, (float)y, 0 ); } ).value();
     ASSERT_TRUE( m.topology.checkValidity() );
}

}