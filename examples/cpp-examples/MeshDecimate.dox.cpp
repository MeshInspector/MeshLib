/// \page ExampleCppMeshDecimate Mesh decimation
/// 
/// Example of mesh decimate
///
/// \code
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshDecimate.h"

int main()
{
    // Load mesh
    MR::Mesh mesh = *MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );

    // Setup decimate parameters
    MR::DecimateSettings settings;
    settings.maxError = 0.05f;

    // Decimate mesh
    MR::decimateMesh( mesh, settings );

    // Save result
    MR::MeshSave::toAnySupportedFormat( mesh, "decimatedMesh.stl" );
}
/// \endcode
