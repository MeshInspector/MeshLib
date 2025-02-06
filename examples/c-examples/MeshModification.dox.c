#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshRelax.h>
#include <MRMeshC/MRMeshSubdivide.h>
#include <MRMeshC/MRTorus.h>

#include <math.h>
#include <stdlib.h>

int main( int argc, char* argv[] )
{
    MRMakeTorusParameters makeTorusParameters = mrMakeTorusParametersNew();
    MRMesh* mesh = mrMakeTorus( &makeTorusParameters );

    // Relax mesh (5 iterations)
    MRRelaxParams relaxParams = mrRelaxParamsNew();
    relaxParams.iterations = 5;
    mrRelax( mesh, &relaxParams, NULL );

    // Subdivide mesh
    MRSubdivideSettings subdivideSettings = mrSubdivideSettingsNew();
    subdivideSettings.maxDeviationAfterFlip = 0.5f;
    mrSubdivideMesh( mesh, &subdivideSettings );

    // Rotate mesh
    MRVector3f plusZ = mrVector3fPlusZ();
    MRMatrix3f rotMat = mrMatrix3fRotationScalar( &plusZ, M_PI_2 );
    MRAffineXf3f rotXf = mrAffineXf3fLinear( &rotMat );
    mrMeshTransform( mesh, &rotXf, NULL );

    mrMeshFree( mesh );
    return EXIT_SUCCESS;
}
