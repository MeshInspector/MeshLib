#include "MRRegularGridMesh.h"
#include "MRMeshBuilder.h"
#include "MRTriMath.h"
#include "MRVector2.h"
#include "MRMeshDelone.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include "MRGTest.h"

namespace MR
{

Mesh makeRegularGridMesh( size_t width, size_t height,
                          const RegularGridLatticeValidator& validator, 
                          const RegularGridLatticePositioner& positioner,
                          const RegularGridMeshFaceValidator& faceValidator )
{
    MR_TIMER
    Mesh res;

    MeshTopology::GridSettings gs =
    {
        .dim = Vector2i( (int)width - 1, (int)height - 1 )
    };

    BitSet validGridVerts( width * height );
    gs.vertIds.b.resize( width * height );
    BitSetParallelForAll( validGridVerts, [&]( size_t p )
    {
        auto y = p / width;
        auto x = p - y * width;
        if ( validator( x, y ) )
            validGridVerts.set( p );
        else
            gs.vertIds.b[p] = VertId{};
    } );

    VertId nextVertId{ 0 };
    for ( auto p : validGridVerts )
        gs.vertIds.b[p] = nextVertId++;

    const auto vertSize = size_t( nextVertId );
    gs.vertIds.tsize = vertSize;
    res.points.resize( vertSize );
    BitSetParallelFor( validGridVerts, [&]( size_t p )
    {
        auto y = p / width;
        auto x = p - y * width;
        res.points[gs.vertIds.b[p]] = positioner( x, y );
    } );

    BitSet validLoUpTris( 2 * ( width - 1 ) * ( height - 1 ) );
    BitSet diagonalA( ( width - 1 ) * ( height - 1 ) ); // if one of triangles is valid

    auto getVertId = [&]( Vector2i pos ) -> VertId
    {
        if ( pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height )
            return VertId();
        return gs.vertIds.b[pos.x + pos.y * width];
    };

    gs.faceIds.b.resize( validLoUpTris.size() );
    BitSetParallelForAll( diagonalA, [&]( size_t p )
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
    } );

    FaceId nextFaceId{ 0 };
    for ( auto p : validLoUpTris )
        gs.faceIds.b[p] = nextFaceId++;
    gs.faceIds.tsize = size_t( nextFaceId );

    auto hasEdge = [&]( Vector2i v, MeshTopology::GridSettings::EdgeType et )
    {
        VertId v0;
        if ( et != MeshTopology::GridSettings::EdgeType::DiagonalB )
            v0 = getVertId( v );
        else 
            v0 = getVertId( Vector2i{ v.x + 1, v.y } );
        if ( !v0 )
            return false;

        VertId v1;
        switch ( et )
        {
        case MeshTopology::GridSettings::EdgeType::Horizontal:
            v1 = getVertId( Vector2i{ v.x + 1, v.y } );
            break;
        case MeshTopology::GridSettings::EdgeType::DiagonalA:
            v1 = getVertId( Vector2i{ v.x + 1, v.y + 1 } );
            break;
        default:
            v1 = getVertId( Vector2i{ v.x, v.y + 1 } );
        }
        if ( !v1 )
            return false;

        if ( v.y + 1 == height )
        {
            assert ( et == MeshTopology::GridSettings::EdgeType::Horizontal );
            assert ( v.x + 1 < width );
            return true;
        }
        if ( v.x + 1 == width && et != MeshTopology::GridSettings::EdgeType::DiagonalB )
        {
            assert ( et == MeshTopology::GridSettings::EdgeType::Vertical );
            return true;
        }
        if ( et == MeshTopology::GridSettings::EdgeType::Horizontal )
            return true;
        if ( et == MeshTopology::GridSettings::EdgeType::Vertical )
            return true;
        auto hfidx = ( width - 1 ) * v.y + v.x;
        if ( !validLoUpTris.test( 2 * hfidx ) && !validLoUpTris.test( 2 * hfidx + 1 ) )
            return false;
        return diagonalA.test( hfidx ) == ( et == MeshTopology::GridSettings::EdgeType::DiagonalA );
    };

    BitSet validGridEdges( 4 * width * height );
    gs.uedgeIds.b.resize( 4 * width * height );
    BitSetParallelForAll( validGridEdges, [&]( size_t loc )
    {
        auto p = loc / 4;
        auto et = MeshTopology::GridSettings::EdgeType( loc - p * 4 );
        auto y = int( p / width );
        auto x = int( p - y * width );
        if ( hasEdge( { x, y }, et ) )
            validGridEdges.set( loc );
        else
            gs.uedgeIds.b[loc] = UndirectedEdgeId{};
    } );

    UndirectedEdgeId nextUEdgeId{ 0 };
    for ( auto p : validGridEdges )
        gs.uedgeIds.b[p] = nextUEdgeId++;
    gs.uedgeIds.tsize = size_t( nextUEdgeId );

    res.topology.buildGridMesh( gs );
    assert( res.topology.checkValidity() );
    return res;
}

TEST(MRMesh, makeRegularGridMesh)
{
     auto m = makeRegularGridMesh( 2, 2,
         []( size_t, size_t ) { return true; },
         []( size_t x, size_t y ) { return Vector3f( (float)x, (float)y, 0 ); } );
     ASSERT_TRUE( m.topology.checkValidity() );

     m = makeRegularGridMesh( 2, 3,
         []( size_t, size_t ) { return true; },
         []( size_t x, size_t y ) { return Vector3f( (float)x, (float)y, 0 ); } );
     ASSERT_TRUE( m.topology.checkValidity() );

     m = makeRegularGridMesh( 5, 3,
         []( size_t, size_t ) { return true; },
         []( size_t x, size_t y ) { return Vector3f( (float)x, (float)y, 0 ); } );
     ASSERT_TRUE( m.topology.checkValidity() );
}

}