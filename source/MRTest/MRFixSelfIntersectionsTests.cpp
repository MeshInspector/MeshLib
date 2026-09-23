#include <MRMesh/MRFixSelfIntersections.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRBitSet.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, FixSelfIntersections )
{
    Mesh mesh = makeTorusWithSelfIntersections( 1.0f, 0.2f, 32, 16 );
    EXPECT_EQ( mesh.topology.getValidFaces().count(), 1024 );

    auto intersections = SelfIntersections::getFaces( mesh, false );
    EXPECT_TRUE( intersections.has_value() );
    EXPECT_EQ( intersections->count(), 128 );

    SelfIntersections::Settings settings;
    settings.method = SelfIntersections::Settings::Method::CutAndFill;
    settings.touchIsIntersection = false;
    EXPECT_TRUE( SelfIntersections::fix( mesh, settings ).has_value() );

    EXPECT_TRUE( mesh.topology.getValidFaces().count() == 1194
              || mesh.topology.getValidFaces().count() == 1196 ); //on some macOS Arm runners in Debug mode

    intersections = SelfIntersections::getFaces( mesh, false );
    EXPECT_TRUE( intersections.has_value() );
    EXPECT_EQ( intersections->count(), 0 );
}

TEST( MRMesh, CutAndFillSelfIntersectionGroups )
{
    Mesh mesh = makeCube();
    FaceBitSet faces( mesh.topology.faceSize() );
    for ( auto f : mesh.topology.getValidFaces() )
    {
        const auto n = mesh.normal( f );
        if ( n.z > 0.9f || n.x > 0.9f )
            faces.set( f );
    }
    EXPECT_EQ( faces.count(), 4 );

    auto [groupsMap, numGroups] = SelfIntersections::getGroupsMap( { mesh, &faces } );
    EXPECT_EQ( numGroups, 2 );
    int groupSize[2] = {};
    for ( auto f : faces )
        ++groupSize[int( groupsMap[f] )];
    EXPECT_EQ( groupSize[0], 2 );
    EXPECT_EQ( groupSize[1], 2 );

    // whole cube: one group per side
    EXPECT_EQ( SelfIntersections::getGroupsMap( { mesh } ).second, 6 );

    // each planar side is refilled separately, so the cube keeps its shape
    auto newFaces = SelfIntersections::cutAndFillGroups( mesh, faces );
    EXPECT_EQ( newFaces.count(), 4 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 12 );
    EXPECT_TRUE( mesh.topology.isClosed() );
    EXPECT_NEAR( mesh.volume(), 1.0, 1e-5 );
}

} //namespace MR
