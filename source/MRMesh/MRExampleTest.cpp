#include "MRMesh/MRMesh.h"
#include "MRMesh/MRTorus.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMeshRelax.h"
#include "MRMesh/MRMeshSubdivide.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRGTest.h"
#include "MRCube.h"

namespace MR
{

TEST( MRMesh, BasicExample )
{
    // load mesh
    MR::Mesh mesh = MR::makeTorus();

    // relax mesh (5 iterations)
    MR::relax( mesh, { {5} } );

    // subdivide mesh
    MR::SubdivideSettings props;
    props.maxDeviationAfterFlip = 0.5f;
    MR::subdivideMesh( mesh, props );

    // rotate mesh
    MR::AffineXf3f rotationXf = MR::AffineXf3f::linear( MR::Matrix3f::rotation( MR::Vector3f::plusZ(), MR::PI_F * 0.5f ) );
    mesh.transform( rotationXf );
}

TEST( MRMesh, PrimitivesExtractionExapmle )
{
    // create some mesh
    MR::Mesh mesh = MR::makeCube();

    // all vertices of valid triangles
    const std::vector<std::array<MR::VertId, 3>> triangles = mesh.topology.getAllTriVerts();

    // all point coordinates
    const std::vector<MR::Vector3f>& points = mesh.points.vec_;
    // triangle vertices as tripples of ints (pointing to elements in points vector)
    const int* vertexTripples = reinterpret_cast< const int* >( triangles.data() );
    ASSERT_EQ( points.size(), 8 );
    ASSERT_NE( vertexTripples, nullptr );
    ASSERT_EQ( triangles.size(), 12 );
}

} // namespace MR
