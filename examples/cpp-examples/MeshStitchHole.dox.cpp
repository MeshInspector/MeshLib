#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshFillHole.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

int main()
{
    // Load meshes
    auto meshARes = MR::MeshLoad::fromAnySupportedFormat( "meshAwithHole.stl" );
    auto meshBRes = MR::MeshLoad::fromAnySupportedFormat( "meshBwithHole.stl" );

    // Unite meshes
    MR::Mesh mesh = std::move( meshARes.value() );
    mesh.addPart( meshBRes.value() );

    // Find holes (expect that there are exactly 2 holes)
    std::vector<MR::EdgeId> edges = mesh.topology.findHoleRepresentiveEdges();
    if ( edges.size() != 2 )
        return 1;

    // Connect two holes
    MR::StitchHolesParams params;
    params.metric = MR::getUniversalMetric( mesh );
    MR::buildCylinderBetweenTwoHoles( mesh, edges.front(), edges.back(), params );

    // Save result
    auto saveRes = MR::MeshSave::toAnySupportedFormat( mesh, "stitchedMesh.stl" );

    return 0;
}
