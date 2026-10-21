#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>

int main()
{
    // create some mesh
    MR::Mesh mesh = MR::makeCube();

    // all vertices of valid triangles
    const std::vector<std::array<MR::VertId, 3>> triangles = mesh.topology.getAllTriVerts();

    // all point coordinates
    const MR::VertCoords & points = mesh.points;
    // triangle vertices as triples of ints (pointing to elements in points vector)
    const int * vertexTriples = reinterpret_cast<const int*>( triangles.data() );

    // TODO: export in your format
    std::ignore = points;
    std::ignore = vertexTriples;
}
