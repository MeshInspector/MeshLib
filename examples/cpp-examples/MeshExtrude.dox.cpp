#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRBitSetParallelFor.h>
#include <MRMesh/MRId.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshExtrude.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRRegionBoundary.h>

int main()
{
    // Load mesh
    MR::Mesh mesh = *MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );

    // Select faces to extrude
    MR::FaceBitSet facesToExtrude;
    facesToExtrude.autoResizeSet( MR::FaceId( 1 ) );
    facesToExtrude.autoResizeSet( MR::FaceId( 2 ) );

    // Create duplicated verts on region boundary
    MR::makeDegenerateBandAroundRegion( mesh, facesToExtrude );

    // Find vertices that will be moved
    auto vertsForMove = MR::getIncidentVerts( mesh.topology, facesToExtrude );

    MR::BitSetParallelFor( vertsForMove, [&] ( MR::VertId v )
    {
        // Move each vertex
        mesh.points[v] += MR::Vector3f::plusZ();
    } );

    // Invalidate internal caches after manual changing
    mesh.invalidateCaches();

    // Save mesh
    MR::MeshSave::toAnySupportedFormat( mesh, "extrudedMesh.stl" );
}
