#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshTopology.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRId.h>
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, findLeftBoundary)
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    FaceBitSet faces;
    faces.autoResizeSet( 0_f );
    auto paths = findLeftBoundary( sphere.topology, faces );
    EXPECT_EQ( paths.size(), 1 );
    for ( const auto & path : paths )
    {
        for ( auto e : path )
        {
            EXPECT_EQ( sphere.topology.left( e ), 0_f );
            EXPECT_NE( sphere.topology.right( e ), 0_f );
        }
    }
}

TEST( MRMesh, findRightBoundary )
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    FaceBitSet faces;
    faces.autoResizeSet( 0_f );
    auto paths = findRightBoundary( sphere.topology, faces );
    EXPECT_EQ( paths.size(), 1 );
    for ( const auto& path : paths )
    {
        for ( auto e : path )
        {
            EXPECT_EQ( sphere.topology.right( e ), 0_f );
            EXPECT_NE( sphere.topology.left( e ), 0_f );
        }
    }
}

} //namespace MR
