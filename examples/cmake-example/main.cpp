
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMeshRelax.h"
#include "MRMesh/MRMeshSubdivide.h"
#include "MRMesh/MRAffineXf3.h"

int main()
{
    MR::Mesh mesh = MR::MeshLoad::fromAnySupportedFormat( "/home/user/CLionProjects/untitled/Torus0.stl" ).value();

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