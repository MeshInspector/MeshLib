#include "MRRegularGridMesh.h"
#include "MRMeshBuilder.h"
#include "MRTriMath.h"
#include "MRVector2.h"
#include "MRMeshDelone.h"
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
    res.points.resize( width * height );

    BitSet validGridVerts( width * height );

    for ( int y = 0; y < height; y++ )
    {
        for ( int x = 0; x < width; x++ )
        {
            auto vidx = width * y + x;
            if ( !validator( x, y ) )
                continue;
            res.points[VertId( vidx )] = positioner( x, y );
            validGridVerts.set( vidx );
        }
    }

    auto getVertId = [&]( Vector2i v )
    {
        if ( v.x < 0 || v.x >= width || v.y < 0 || v.y >= height )
            return VertId();
        auto vidx = width * v.y + v.x;
        return validGridVerts.test( vidx ) ? VertId( vidx ) : VertId();
    };

    BitSet validLoUpTris( 2 * ( width - 1 ) * ( height - 1 ) );
    BitSet diagonalA( ( width - 1 ) * ( height - 1 ) ); // if one of triangles is valid

    for ( int y = 0; y + 1 < height; y++ )
    {
        for ( int x = 0; x + 1 < width; x++ )
        {
            auto hfidx = ( width - 1 ) * y + x;

            const Vector2i p00{ x, y };
            const Vector2i p01{ x, y + 1 };
            const Vector2i p10{ x + 1, y };
            const Vector2i p11{ x + 1, y + 1 };

            const auto v00 = getVertId( p00 );
            const auto v01 = getVertId( p01 );
            const auto v10 = getVertId( p10 );
            const auto v11 = getVertId( p11 );
            const auto count = v00.valid() + v01.valid() + v10.valid() + v11.valid();

            auto canCreateFace = [&]( Vector2i a, Vector2i b, Vector2i c )
            {
                if ( !faceValidator )
                    return true;
                return faceValidator( a.x, a.y, b.x, b.y, c.x, c.y );
            };

            switch ( count )
            {
            case 4:
                // two possible triangles
                if ( checkDeloneQuadrangle( res.points[v00], res.points[v01], res.points[v11], res.points[v10] ) )
                {
                    diagonalA.set( hfidx );
                    if ( canCreateFace( p11, p01, p00 ) )
                        validLoUpTris.set( 2 * hfidx + 1 ); //upper
                    if ( canCreateFace( p11, p00, p10 ) )
                        validLoUpTris.set( 2 * hfidx ); //lower
                }
                else
                {
                    if ( canCreateFace( p01, p00, p10 ) )
                        validLoUpTris.set( 2 * hfidx ); //lower
                    if ( canCreateFace( p01, p10, p11 ) )
                        validLoUpTris.set( 2 * hfidx + 1 ); //upper
                }
                break;
            case 3:
                // one possible triangle
                if ( !v00 )
                {
                    if ( canCreateFace( p01, p10, p11 ) )
                        validLoUpTris.set( 2 * hfidx + 1 ); //upper
                }
                else if ( !v01 )
                {
                    diagonalA.set( hfidx );
                    if ( canCreateFace( p11, p00, p10 ) )
                        validLoUpTris.set( 2 * hfidx ); //lower
                }
                else if ( !v10 )
                {
                    diagonalA.set( hfidx );
                    if ( canCreateFace( p11, p01, p00 ) )
                        validLoUpTris.set( 2 * hfidx + 1 ); //upper
                }
                else if ( !v11 )
                {
                    if ( canCreateFace( p01, p00, p10 ) )
                        validLoUpTris.set( 2 * hfidx ); //lower
                }
                break;
            case 2:
            case 1:
            case 0:
                // no possible triangles
                break;
            }
        }
    }

    auto getFaceId = [&]( Vector2i v, MeshTopology::GridSettings::TriType tt )
    {
        if ( v.x < 0 || v.x + 1 >= width || v.y < 0 || v.y + 1 >= height )
            return FaceId();
        auto fidx = 2 * ( ( width - 1 ) * v.y + v.x );
        if ( tt == MeshTopology::GridSettings::TriType::Upper )
            ++fidx;
        return validLoUpTris.test( fidx ) ? FaceId( fidx ) : FaceId();
    };

    auto getEdgeId = [&]( Vector2i v, MeshTopology::GridSettings::EdgeType et )
    {
        VertId v0;
        if ( et != MeshTopology::GridSettings::EdgeType::DiagonalB )
            v0 = getVertId( v );
        else 
            v0 = getVertId( Vector2i{ v.x + 1, v.y } );
        if ( !v0 )
            return EdgeId();

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
            return EdgeId();

        auto ueidx = ( 3 * width - 2 ) * v.y;
        if ( v.y + 1 == height )
        {
            ueidx += v.x;
            assert ( et == MeshTopology::GridSettings::EdgeType::Horizontal );
            assert ( v.x + 1 < width );
            return EdgeId( 2 * ueidx );
        }
        ueidx += 3 * v.x;
        if ( v.x + 1 == width && et != MeshTopology::GridSettings::EdgeType::DiagonalB )
        {
            assert ( et == MeshTopology::GridSettings::EdgeType::Vertical );
            return EdgeId( 2 * ueidx );
        }
        if ( et == MeshTopology::GridSettings::EdgeType::Horizontal )
            return EdgeId( 2 * ueidx );
        if ( et == MeshTopology::GridSettings::EdgeType::Vertical )
            return EdgeId( 2 * ( ueidx + 1 ) );
        auto hfidx = ( width - 1 ) * v.y + v.x;
        if ( !validLoUpTris.test( 2 * hfidx ) && !validLoUpTris.test( 2 * hfidx + 1 ) )
            return EdgeId();
        return diagonalA.test( hfidx ) == ( et == MeshTopology::GridSettings::EdgeType::DiagonalA ) ? EdgeId( 2 * ( ueidx + 2 ) ) : EdgeId();
    };

    MeshTopology::GridSettings gs =
    {
        .dim = Vector2i( (int)width - 1, (int)height - 1),
        .getVertId = getVertId,
        .getEdgeId = getEdgeId,
        .getFaceId = getFaceId
    };
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