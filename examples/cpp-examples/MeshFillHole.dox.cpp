#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshFillHole.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

int main()
{
    // Load mesh
    MR::Mesh mesh = *MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );

    // Find single edge for each hole in mesh
    std::vector<MR::EdgeId> holeEdges = mesh.topology.findHoleRepresentiveEdges();

    for ( MR::EdgeId e : holeEdges )
    {
        // Setup filling parameters
        MR::FillHoleParams params;
        params.metric = MR::getUniversalMetric( mesh );
        // Fill hole represented by `e`
        MR::fillHole( mesh, e, params );
    }

    // Save result
    auto saveRes = MR::MeshSave::toAnySupportedFormat( mesh, "filledMesh.stl" );
}
