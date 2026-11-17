#include <MRMesh/MRPointsToMeshProjector.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshProject.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRVector.h>
#include <gtest/gtest.h>

namespace MR
{

// the reference sphere of radius 1 is stretched twice along x-axis by non-rigid refXf
TEST( MRMesh, PointsToMeshProjectorNonRigidRefXf )
{
    const Mesh sphere = makeUVSphere( 1.0f, 64, 64 );
    const AffineXf3f refXf( Matrix3f::scale( 2.0f, 1.0f, 1.0f ), Vector3f( 1.0f, 0.0f, 0.0f ) );

    // world points: outside and inside the stretched sphere (centered at x=1, half-axis 2 along x)
    VertCoords pts;
    pts.push_back( Vector3f( 3.5f, 0.0f, 0.0f ) );  // outside, distance 0.5
    pts.push_back( Vector3f( 2.5f, 0.0f, 0.0f ) );  // inside, distance 0.5
    pts.push_back( Vector3f( 1.0f, 1.25f, 0.0f ) ); // outside, distance 0.25
    pts.push_back( Vector3f( 1.0f, 0.0f, -0.5f ) ); // inside, distance 0.5
    const float expected[] = { 0.5f, -0.5f, 0.25f, -0.5f };

    // projector returns world distances and the points in the space of the reference mesh
    PointsToMeshProjector projector;
    projector.updateMeshData( &sphere );
    std::vector<MeshProjectionResult> projs;
    projector.findProjections( projs, pts, nullptr, &refXf, FLT_MAX, 0 );
    ASSERT_EQ( projs.size(), pts.size() );
    for ( size_t i = 0; i < projs.size(); ++i )
    {
        const auto & p = projs[i];
        ASSERT_TRUE( p.valid() );
        EXPECT_NEAR( std::sqrt( p.distSq ), std::abs( expected[i] ), 0.01f );
        EXPECT_NEAR( ( refXf( p.proj.point ) - pts[VertId( i )] ).length(), std::abs( expected[i] ), 0.01f );
        EXPECT_NEAR( ( sphere.triPoint( p.mtp( sphere.topology ) ) - p.proj.point ).length(), 0.0f, 1e-5f );
    }

    // signed distances are the same with and without the projector, and have correct signs
    MeshProjectionParameters params;
    params.refXf = &refXf;
    const auto withProjector = findSignedDistances( sphere, pts, nullptr, params, &projector );
    const auto withoutProjector = findSignedDistances( sphere, pts, nullptr, params );
    for ( size_t i = 0; i < pts.size(); ++i )
    {
        EXPECT_NEAR( withProjector[VertId( i )], expected[i], 0.01f );
        EXPECT_NEAR( withoutProjector[VertId( i )], expected[i], 0.01f );
    }
}

} //namespace MR
