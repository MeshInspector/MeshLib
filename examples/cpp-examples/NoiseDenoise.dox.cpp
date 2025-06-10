#include <MRMesh/MRAddNoise.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRNormalDenoising.h>

#include <iostream>

int main()
{
    // Load mesh
    auto loadRes = MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );
    if ( !loadRes.has_value() )
    {
        std::cerr << loadRes.error() << std::endl;
        return 1;
    }
    MR::Mesh& mesh = *loadRes;

    // Add noise to the mesh
    std::ignore = MR::addNoise( mesh.points, mesh.topology.getValidVerts(), {
        .sigma = mesh.computeBoundingBox().diagonal() * 0.0001f,
    } );

    // Invalidate the mesh because of the external vertex changes
    mesh.invalidateCaches();

    // Save the noised mesh
    if ( auto saveRes = MR::MeshSave::toAnySupportedFormat( mesh, "noised_mesh.stl" ); !saveRes )
    {
        std::cerr << saveRes.error() << std::endl;
        return 1;
    }

    // Denoise the mesh with sharpening for sharp edges
    // see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
    std::ignore = MR::meshDenoiseViaNormals( mesh );

    // Save the denoised mesh
    if ( auto saveRes = MR::MeshSave::toAnySupportedFormat( mesh, "denoised_mesh.stl" ); !saveRes )
    {
        std::cerr << saveRes.error() << std::endl;
        return 1;
    }
}
