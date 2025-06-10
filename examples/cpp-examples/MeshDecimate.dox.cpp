#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshDecimate.h>
#include <MRMesh/MRBuffer.h>

#include <iostream>

int main()
{
    // Load mesh
    auto meshRes = MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );
    if ( !meshRes.has_value() )
    {
        std::cerr << meshRes.error() << std::endl;
        return 1;
    }

    MR::Mesh& mesh = *meshRes;
    
//! [0]
    // Repack mesh optimally.
    // It's not necessary but highly recommended to achieve the best performance in parallel processing
    mesh.packOptimally();

    // Setup decimate parameters
    MR::DecimateSettings settings;

    // Decimation stop thresholds, you may specify one or both
    settings.maxDeletedFaces = 1000; // Number of faces to be deleted
    settings.maxError = 0.05f; // Maximum error when decimation stops

    // Number of parts to simultaneous processing, greatly improves performance by cost of minor quality loss.
    // Recommended to set to the number of available CPU cores or more for the best performance
    settings.subdivideParts = 64;

    // Decimate mesh
    MR::decimateMesh( mesh, settings );
//! [0]

    // Save result
    if ( auto saveRes = MR::MeshSave::toAnySupportedFormat( mesh, "decimated_mesh.stl" ); !saveRes )
    {
        std::cerr << saveRes.error() << std::endl;
        return 1;
    }

    return 0;
}
