#include <gtest/gtest.h>
#include <MRMesh/MRCutAroundEdgePaths.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MREdgePaths.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRSurfaceDistance.h>

namespace MR
{

TEST( MRMesh, CutAroundEdgePaths )
{
    auto mesh = makeSphere( { .radius = 1, .numMeshVertices = 4000 } );

    auto closestVert = [&]( const Vector3f & p )
    {
        VertId res;
        float bestDistSq = FLT_MAX;
        for ( auto v : mesh.topology.getValidVerts() )
        {
            const auto distSq = ( mesh.points[v] - p ).lengthSq();
            if ( distSq < bestDistSq )
            {
                bestDistSq = distSq;
                res = v;
            }
        }
        return res;
    };

    // two meridian-like paths about 0.36 apart, closer than 2 * distance + minSpacing
    const std::vector<EdgePath> paths =
    {
        buildShortestPath( mesh, closestVert( Vector3f( 1, -0.2f, 0.5f ).normalized() ), closestVert( Vector3f( 1, -0.2f, -0.5f ).normalized() ) ),
        buildShortestPath( mesh, closestVert( Vector3f( 1, 0.2f, 0.5f ).normalized() ), closestVert( Vector3f( 1, 0.2f, -0.5f ).normalized() ) )
    };
    const auto numFaces0 = mesh.topology.numValidFaces();

    const CutAroundEdgePathsParams params{ .distance = 0.15f, .minSpacing = 0.1f };
    ASSERT_FALSE( paths[0].empty() );
    ASSERT_FALSE( paths[1].empty() );
    auto res = cutAroundEdgePaths( mesh, paths, params );
    ASSERT_TRUE( res.has_value() );
    ASSERT_EQ( res->size(), 2 );
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_GT( mesh.topology.numValidFaces(), numFaces0 );

    const auto & r0 = ( *res )[0];
    const auto & r1 = ( *res )[1];
    EXPECT_TRUE( r0.any() );
    EXPECT_TRUE( r1.any() );
    EXPECT_FALSE( r0.intersects( r1 ) );

    // each path is inside its region
    for ( int i = 0; i < 2; ++i )
        for ( auto e : paths[i] )
            EXPECT_TRUE( ( *res )[i].test( mesh.topology.left( e ) ) );

    // the regions are separated: about 0.36 * minSpacing / ( 2 * distance + minSpacing ) = 0.09, and only 0.02 if minSpacing = 0
    const auto verts0 = getIncidentVerts( mesh.topology, r0 );
    const auto verts1 = getIncidentVerts( mesh.topology, r1 );
    EXPECT_FALSE( verts0.intersects( verts1 ) );
    const auto distFrom0 = computeSurfaceDistances( mesh, verts0, 1.0f );
    float minDist = FLT_MAX;
    for ( auto v : verts1 )
        minDist = std::min( minDist, distFrom0[v] );
    EXPECT_GT( minDist, 0.5f * params.minSpacing );
}

} //namespace MR
