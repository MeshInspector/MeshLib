#include <MRMesh/MRBox.h>
#include <MRMesh/MRICP.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

#include <iostream>

int main()
{
    // Load meshes
    MR::Mesh meshFloating = *MR::MeshLoad::fromAnySupportedFormat( "meshA.stl" );
    MR::Mesh meshFixed = *MR::MeshLoad::fromAnySupportedFormat( "meshB.stl" );

    // Prepare ICP parameters
    float diagonal = meshFixed.getBoundingBox().diagonal();
    float icpSamplingVoxelSize = diagonal * 0.01f; // To sample points from object
    MR::ICPProperties icpParams;
    icpParams.distThresholdSq = MR::sqr( diagonal * 0.1f ); // Use points pairs with maximum distance specified
    icpParams.exitVal = diagonal * 0.003f; // Stop when distance reached

    // Calculate transformation
    MR::ICP icp(
        MR::MeshOrPoints{ MR::MeshPart{ meshFloating } },
        MR::MeshOrPoints{ MR::MeshPart{ meshFixed } },
        MR::AffineXf3f(), MR::AffineXf3f(),
        icpSamplingVoxelSize );
    icp.setParams( icpParams );
    MR::AffineXf3f xf = icp.calculateTransformation();

    // Transform floating mesh
    meshFloating.transform( xf );

    // Output information string
    std::string info = icp.getStatusInfo();
    std::cerr << info << std::endl;

    // Save result
    MR::MeshSave::toAnySupportedFormat( meshFloating, "meshA_icp.stl" );
}
