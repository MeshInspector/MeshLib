#include <MRCMesh/MRAffineXf.h>
#include <MRCMesh/MRMatrix3.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshRelax.h>
#include <MRCMesh/MRMeshSubdivide.h>
#include <MRCMesh/MRTorus.h>

#include <math.h>
#include <stdlib.h>

int main( void )
{
    MR_Mesh* mesh = MR_makeTorus( NULL, NULL, NULL, NULL, NULL );

    // Relax mesh (5 iterations)
    MR_MeshRelaxParams* meshRelaxParams = MR_MeshRelaxParams_DefaultConstruct();
    MR_RelaxParams* relaxParams = MR_MeshRelaxParams_MutableUpcastTo_MR_RelaxParams( meshRelaxParams );
    MR_RelaxParams_Set_iterations( relaxParams, 5 );
    MR_relax_3_MR_Mesh( mesh, meshRelaxParams, NULL );
    MR_MeshRelaxParams_Destroy( meshRelaxParams );

    // Subdivide mesh
    MR_SubdivideSettings* subdivideSettings = MR_SubdivideSettings_DefaultConstruct();
    MR_SubdivideSettings_Set_maxDeviationAfterFlip( subdivideSettings, 0.5f );
    MR_subdivideMesh_MR_Mesh( mesh, subdivideSettings );
    MR_SubdivideSettings_Destroy( subdivideSettings );

    // Rotate mesh
    MR_Vector3f plusZ = MR_Vector3f_plusZ();
    MR_Matrix3f rotMat = MR_Matrix3f_rotation_float( &plusZ, 1.57079632679489661923 ); // pi/2
    MR_AffineXf3f rotXf = MR_AffineXf3f_linear( &rotMat );
    MR_Mesh_transform( mesh, &rotXf, NULL );

    MR_Mesh_Destroy( mesh );
    return EXIT_SUCCESS;
}
