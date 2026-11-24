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

TEST( MRMesh, BuildSmallestMetricPathBiDirTriPoints )
{
    Mesh cube = makeCube();
    const auto & topology = cube.topology;
    const MeshTriPoint start( topology, 0_v ), finish( topology, 6_v );
    const auto euclid = edgeLengthMetric( cube );
    VertId s, f;
    const auto path = buildSmallestMetricPathBiDir( cube, euclid, start, finish, &s, &f );
    ASSERT_EQ( path.size(), 2 );
    EXPECT_EQ( s, 0_v );
    EXPECT_EQ( f, 6_v );

    // the edges with infinite metric are never passed
    const auto mid = topology.dest( path[0] );
    const auto restricted = buildSmallestMetricPathBiDir( cube, [&]( EdgeId e )
        { return topology.org( e ) == mid || topology.dest( e ) == mid ? FLT_MAX : euclid( e ); }, start, finish );
    ASSERT_FALSE( restricted.empty() );
    EXPECT_EQ( topology.org( restricted.front() ), 0_v );
    EXPECT_EQ( topology.dest( restricted.back() ), 6_v );
    for ( size_t i = 0; i < restricted.size(); ++i )
    {
        EXPECT_NE( topology.dest( restricted[i] ), mid );
        if ( i + 1 < restricted.size() )
        {
            EXPECT_EQ( topology.dest( restricted[i] ), topology.org( restricted[i + 1] ) );
        }
    }
    EXPECT_GE( calcPathMetric( restricted, euclid ), calcPathMetric( path, euclid ) );
}

} //namespace MR
