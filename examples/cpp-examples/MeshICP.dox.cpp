#include <MRMesh/MRBox.h>
#include <MRMesh/MRICP.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

#include <iostream>

int main()
{
    // Load meshes
    auto meshFloatingRes = MR::MeshLoad::fromAnySupportedFormat( "meshA.stl" );
    if ( !meshFloatingRes.has_value() )
    {
        std::cerr << meshFloatingRes.error() << std::endl;
        return 1;
    }
    MR::Mesh& meshFloating = *meshFloatingRes;

    auto meshFixedRes = MR::MeshLoad::fromAnySupportedFormat( "meshB.stl" );
    if ( !meshFixedRes.has_value() )
    {
        std::cerr << meshFixedRes.error() << std::endl;
        return 1;
    }
    MR::Mesh& meshFixed = *meshFixedRes;

//! [0]
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
//! [0]

    // Output information string
    std::string info = icp.getStatusInfo();
    std::cerr << info << std::endl;

    // Save result
    if ( auto saveRes = MR::MeshSave::toAnySupportedFormat( meshFloating, "meshA_icp.stl" ); !saveRes )
    {
        std::cerr << saveRes.error() << std::endl;
        return 1;
    }
}
