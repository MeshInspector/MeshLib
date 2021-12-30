#include "MRMakePlane.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"

namespace MR
{

Mesh makePlane()
{
    Mesh plane;
    plane.points.resize( 4 );
    plane.points[VertId( 0 )] = Vector3f{-0.5f,-0.5f,0.0f};
    plane.points[VertId( 1 )] = Vector3f{-0.5f,0.5f,0.0f};
    plane.points[VertId( 2 )] = Vector3f{0.5f,0.5f,0.0f};
    plane.points[VertId( 3 )] = Vector3f{0.5f,-0.5f,0.0f};

    std::vector<MeshBuilder::Triangle> tris(2);
    tris[0] = {VertId( 2 ),VertId( 1 ),VertId( 0 ),FaceId( 0 )};
    tris[1] = {VertId( 0 ),VertId( 3 ),VertId( 2 ),FaceId( 1 )};
    plane.topology = MeshBuilder::fromTriangles( tris );
    return plane;
}

}
