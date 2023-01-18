#include "MRRegularGridMesh.h"
#include "MRMeshBuilder.h"
#include "MRTriMath.h"
#include "MRVector2.h"

namespace MR
{

Mesh makeRegularGridMesh( size_t width, size_t height,
                          const RegularGridLatticeValidator& validator, 
                          const RegularGridLatticePositioner& positioner,
                          const RegularGridMeshFaceValidator& faceValidator )
{
    Mesh res;
    res.points.resize( width * height );
    auto isInside = [&] ( VertId v, VertId v0, VertId v1, VertId v2 )
    {
        const Vector2f p( res.points[v] ),
                       p0( res.points[v0] ),
                       p1( res.points[v1] ),
                       p2( res.points[v2] );
        const auto triArea = area( p0, p1, p2 );
        const auto sumArea = area( p, p0, p1 ) + area( p, p1, p2 ) + area( p, p0, p2 );
        return std::abs( triArea - sumArea ) <= FLT_EPSILON;
    };

    Triangulation faces;
    auto addFace = [&] ( VertId v0, VertId v1, VertId v2 )
    {
        if ( faceValidator )
        {
            const auto x0 = v0 % width,
                       x1 = v1 % width,
                       x2 = v2 % width;
            const auto y0 = v0 / width,
                       y1 = v1 / width,
                       y2 = v2 / width;
            if ( !faceValidator( x0, y0, x1, y1, x2, y2 ) )
                return;
        }
        faces.push_back( { v0, v1, v2 } );
    };

    for ( int y = 0; y < height; y++ )
    {
        for ( int x = 0; x < width; x++ )
        {
            auto idx = width * y + x;
            res.points[VertId( idx )] = validator( x, y ) ? positioner( x, y ) : Vector3f();

            if ( y == 0 || x == 0 )
                continue;

            const auto v00 = VertId( idx - 1 - width ); // x - 1, y - 1
            const auto v01 = VertId( idx - 1 );         // x - 1, y
            const auto v10 = VertId( idx - width );     // x, y - 1
            const auto v11 = VertId( idx );             // x, y

            std::bitset<4> validVerts;
            validVerts.set( 0b00, validator( x - 1, y - 1 ) );
            validVerts.set( 0b01, validator( x - 1, y ) );
            validVerts.set( 0b10, validator( x, y - 1 ) );
            validVerts.set( 0b11, validator( x, y ) );

            switch ( validVerts.count() )
            {
            case 4:
                // two possible triangles
                if ( !isInside( v01, v11, v00, v10 ) && !isInside( v10, v11, v01, v00 ) )
                {
                    addFace( v11, v00, v10 );
                    addFace( v11, v01, v00 );
                }
                else
                {
                    addFace( v01, v00, v10 );
                    addFace( v01, v10, v11 );
                }
                break;
            case 3:
                // one possible triangle
                if ( !validVerts.test( 0b00 ) )
                    addFace( v11, v01, v10 );
                else if ( !validVerts.test( 0b01 ) )
                    addFace( v11, v00, v10 );
                else if ( !validVerts.test( 0b10 ) )
                    addFace( v11, v01, v00 );
                else if ( !validVerts.test( 0b11 ) )
                    addFace( v01, v00, v10 );
                break;
            case 2:
            case 1:
            case 0:
                // no possible triangles
                break;
            }
        }
    }
    res.topology = MeshBuilder::fromTriangles( faces );
    return res;
}

}