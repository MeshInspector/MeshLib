#include <MRMesh/MRBox.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshFixer.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

#include <iostream>

int main()
{
    auto mesh = MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );
    if ( !mesh )
    {
        std::cerr << mesh.error() << std::endl;
        return 1;
    }

    // you can set various parameters for the resolving process; see the documentation for more info
    std::ignore = MR::fixMeshDegeneracies( *mesh, {
        .maxDeviation = 1e-5f * mesh->computeBoundingBox().diagonal(),
        .tinyEdgeLength = 1e-3f,
    } );

    // Save result
    if ( auto saveRes = MR::MeshSave::toAnySupportedFormat( *mesh, "fixedMesh.stl" ); !saveRes )
    {
        std::cerr << saveRes.error() << std::endl;
        return 1;
    }

    return 0;
}
