#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshSave.h>

#include <iostream>

int main()
{
    // create a mesh in memory: verifies that headers and libraries match
    const MR::Mesh mesh = MR::makeCube();

    const int verts = mesh.topology.numValidVerts();
    const int faces = mesh.topology.numValidFaces();
    std::cout << "verts=" << verts << " faces=" << faces << std::endl;
    if ( verts != 8 || faces != 12 )
    {
        std::cerr << "unexpected cube topology" << std::endl;
        return 1;
    }

    // write it out: verifies that the library can reach the filesystem
    const auto saveRes = MR::MeshSave::toBinaryStl( mesh, "cube.stl" );
    if ( !saveRes.has_value() )
    {
        std::cerr << saveRes.error() << std::endl;
        return 1;
    }

    std::cout << "MeshLib setup OK" << std::endl;
    return 0;
}
