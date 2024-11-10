#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>

int main()
{
    // create some mesh
    MR::Mesh mesh = MR::makeCube();

    // all vertices of valid triangles
    const std::vector<std::array<MR::VertId, 3>> triangles = mesh.topology.getAllTriVerts();

    // all point coordinates
    const std::vector<MR::Vector3f> & points =  mesh.points.vec_;
    // triangle vertices as tripples of ints (pointing to elements in points vector)
    const int * vertexTripples = reinterpret_cast<const int*>( triangles.data() );

    return 0;
}
