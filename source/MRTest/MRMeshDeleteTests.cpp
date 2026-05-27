#include <MRMesh/MRMeshDelete.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST(MRMesh, DeleteTargetFaces)
{
    Mesh meshObj = makeCube({ 1.f, 1.f, 1.f }, { 0.f, 0.f, 0.f });
    Mesh meshRef = makeCube({ 1.f, 1.f, 1.f }, { -1.f, -1.f, -1.f });

    EXPECT_EQ(meshObj.topology.numValidVerts(), 8);
    EXPECT_EQ(meshObj.topology.numValidFaces(), 12);
    EXPECT_EQ(meshObj.points.size(), 8);

    deleteTargetFaces(meshObj, meshRef);

    EXPECT_EQ(meshObj.topology.numValidVerts(), 7);
    EXPECT_EQ(meshObj.topology.numValidFaces(), 6);
    EXPECT_EQ(meshObj.points.size(), 8);
}

} //namespace MR
