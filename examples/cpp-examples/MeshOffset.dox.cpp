#include <MRMesh/MRBox.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRVoxels/MROffset.h>

int main()
{
    // Load mesh
    MR::Mesh mesh = *MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );

    // Setup parameters
    MR::GeneralOffsetParameters params;
    params.voxelSize = mesh.computeBoundingBox().diagonal() * 5e-3f; // offset grid precision (algorithm is voxel based)
    if ( !MR::findRightBoundary( mesh.topology ).empty() )
        params.signDetectionMode = MR::SignDetectionMode::HoleWindingRule; // use if you have holes in mesh

    // Make offset mesh
    float offset = mesh.computeBoundingBox().diagonal() * 0.05f;
    auto meshRes = MR::generalOffsetMesh( mesh, offset, params );
    if ( !meshRes.has_value() )
    {
        // log meshRes.error()
        return 1;
    }

    // Save result
    MR::MeshSave::toAnySupportedFormat( *meshRes, "mesh_offset.stl" );

    return 0;
}
