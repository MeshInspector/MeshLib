#include "MRMeshLoad.h"
#include "MRMeshSave.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, LoadSave) 
{
    std::string file = 
        "OFF\n"
        "5 6 0\n"

        "0 0 1\n"
        "1 0 0\n"
        "0 1 0\n"
        "-1 0 0\n"
        "0 -1 0\n"

        "3 0 1 2\n"
        "3 0 2 3\n"
        "3 0 3 4\n"
        "3 0 4 1\n"
        "3 1 3 2\n"
        "3 1 4 3\n";
     
    std::istringstream in( file );

    auto loadRes = MeshLoad::fromOff( in );
    EXPECT_TRUE( loadRes.has_value() );

    EXPECT_EQ( loadRes->points.size(), 5 );
    EXPECT_EQ( loadRes->topology.numValidVerts(), 5 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 6 );

    auto box = loadRes->computeBoundingBox();
    EXPECT_EQ( box, Box3f( Vector3f(-1, -1, 0), Vector3f(1, 1, 1) ) );
    EXPECT_TRUE ( box.contains( Vector3f(0, 0, 0) ) );
    EXPECT_FALSE( box.contains( Vector3f(-1, -1, -1) ) );

    std::stringstream ss;
    auto saveRes = MeshSave::toOff( *loadRes, ss );
    EXPECT_TRUE( saveRes.has_value() );

    loadRes = MeshLoad::fromOff( ss );
    EXPECT_TRUE( loadRes.has_value() );

    EXPECT_EQ( loadRes->points.size(), 5 );
    EXPECT_EQ( loadRes->topology.numValidVerts(), 5 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 6 );
    
    // save/load to internal format
    ss = std::stringstream{};
    saveRes = MeshSave::toMrmesh( *loadRes, ss );
    EXPECT_TRUE( saveRes.has_value() );

    loadRes = MeshLoad::fromMrmesh( ss );
    EXPECT_TRUE( loadRes.has_value() );

    EXPECT_EQ( loadRes->points.size(), 5 );
    EXPECT_EQ( loadRes->topology.numValidVerts(), 5 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 6 );

    // save/load to binary STL format
    ss = std::stringstream{};
    saveRes = MeshSave::toBinaryStl( *loadRes, ss );
    EXPECT_TRUE( saveRes.has_value() );

    loadRes = MeshLoad::fromBinaryStl( ss );
    EXPECT_TRUE( loadRes.has_value() );

    EXPECT_EQ( loadRes->points.size(), 5 );
    EXPECT_EQ( loadRes->topology.numValidVerts(), 5 );
    EXPECT_EQ( loadRes->topology.numValidFaces(), 6 );
}

} //namespace MR
