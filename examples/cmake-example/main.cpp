#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshRelax.h"
#include "MRMesh/MRMeshSubdivide.h"
#include "MRMesh/MRTorus.h"

int main()
{
    MR::Mesh mesh = MR::makeTorus();

    // relax mesh (5 iterations)
    MR::relax( mesh, {{5}} );

    // subdivide mesh
    MR::SubdivideSettings props;
    props.maxDeviationAfterFlip = 0.5f;
    MR::subdivideMesh( mesh, props );

    // rotate mesh
    MR::AffineXf3f rotationXf = MR::AffineXf3f::linear( MR::Matrix3f::rotation( MR::Vector3f::plusZ(), MR::PI_F*0.5f ) );
    mesh.transform( rotationXf );

    return 0;
}