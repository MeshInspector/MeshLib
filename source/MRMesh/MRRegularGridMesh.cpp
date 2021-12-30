#include "MRRegularGridMesh.h"
#include "MRMeshBuilder.h"

namespace MR
{

Mesh makeRegularGridMesh( size_t width, size_t height,
                          const RegularGridLatticeValidator& validator, 
                          const RegularGridLatticePositioner& positioner )
{
    Mesh res;
    res.points.resize( width*height );
    std::vector<MeshBuilder::Triangle> faces;
    int faceIdx = 0;
    for ( int y = 0; y < height; y++ )
    {
        for ( int x = 0; x < width; x++ )
        {
            auto idx = width * y + x;
            res.points[VertId( idx )] = validator( x, y ) ? positioner( x, y ) : Vector3f();

            if ( y > 0 && x > 0 )
            {
                // add triangles
                if ( validator( x, y ) )
                {
                    if ( validator( x - 1, y - 1 ) )
                    {
                        if ( validator( x, y - 1 ) )
                            faces.push_back( MeshBuilder::Triangle(
                                VertId( idx ),
                                VertId( idx - 1 - width ),
                                VertId( idx - width ),
                                FaceId( faceIdx++ ) ) );
                        if ( validator( x - 1, y ) )
                            faces.push_back( MeshBuilder::Triangle(
                                VertId( idx ),
                                VertId( idx - 1 ),
                                VertId( idx - 1 - width ),
                                FaceId( faceIdx++ ) ) );
                    }
                    else
                    {
                        if ( validator( x, y - 1 ) && validator( x - 1, y ) )
                            faces.push_back( MeshBuilder::Triangle(
                                VertId( idx ),
                                VertId( idx - 1 ),
                                VertId( idx - width ),
                                FaceId( faceIdx++ ) ) );
                    }
                }
                else
                {
                    if ( validator( x - 1, y - 1 ) && validator(x, y - 1 ) && validator( x - 1, y ) )
                        faces.push_back( MeshBuilder::Triangle(
                            VertId( idx - 1 ),
                            VertId( idx - 1 - width ),
                            VertId( idx - width ),
                            FaceId( faceIdx++ ) ) );
                }
            }
        }
    }
    res.topology = MeshBuilder::fromTriangles( faces );
    return res;
}

}