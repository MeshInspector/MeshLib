#include <MRMesh/MREdgePaths.h>
#include <MRMesh/MREdgeMetric.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <gtest/gtest.h>

namespace MR
{

TEST(MRMesh, BuildShortestPath)
{
    Mesh cube = makeCube();
    auto path = buildShortestPath( cube, 0_v, 6_v );
    EXPECT_EQ( path.size(), 2 );
    EXPECT_EQ( cube.topology.org( path[0] ), 0_v );
    EXPECT_EQ( cube.topology.dest( path[0] ), cube.topology.org( path[1] ) );
    EXPECT_EQ( cube.topology.dest( path[1] ), 6_v );

    auto path34 = buildShortestPath( cube, 3_v, 4_v );
    EXPECT_EQ( path34.size(), 2 );

    std::vector<EdgePath> paths{ path, path34 };
    auto euclid = edgeLengthMetric( cube );
    EXPECT_GT( calcPathMetric( paths[0], euclid ), calcPathMetric( paths[1], euclid ) );
    sortPathsByMetric( paths, euclid );
    EXPECT_LE( calcPathMetric( paths[0], euclid ), calcPathMetric( paths[1], euclid ) );
}

} //namespace MR
