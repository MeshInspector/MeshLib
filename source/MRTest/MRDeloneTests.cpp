#include "MRMesh/MRMeshDelone.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRVector3.h"
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, checkDeloneQuadrangle )
{
    Vector3d a( 0, 0, 0 );
    Vector3d b( 1, 0, 0 );
    Vector3d c( 2, 0, 0 );
    Vector3d d( 1.1, 0, 0 );
    // ABCD quadrangle has zero area
    EXPECT_FALSE( checkDeloneQuadrangle( a, b, c, d, 10 ) );
}

TEST( MRMesh, DeloneFlipsSmallMeshOneIter )
{
    // several separate thin rhombi, each triangulated across its long diagonal (not Delone); the sides are
    // boundary edges, so one iteration must flip exactly one edge per rhombus whatever the edge order is
    constexpr int cNumRhombi = 4;
    VertCoords points;
    Triangulation t;
    for ( int i = 0; i < cNumRhombi; ++i )
    {
        const float x = 10.0f * i;
        const VertId v0( int( points.size() ) );
        points.push_back( Vector3f( x - 1, 0, 0 ) );
        points.push_back( Vector3f( x, -0.2f, 0 ) );
        points.push_back( Vector3f( x + 1, 0, 0 ) );
        points.push_back( Vector3f( x, 0.2f, 0 ) );
        t.push_back( ThreeVertIds{ v0, v0 + 1, v0 + 2 } );
        t.push_back( ThreeVertIds{ v0, v0 + 2, v0 + 3 } );
    }
    Mesh mesh = Mesh::fromTriangles( std::move( points ), t );
    EXPECT_EQ( makeDeloneEdgeFlips( mesh, {}, 1 ), cNumRhombi );
    EXPECT_EQ( makeDeloneEdgeFlips( mesh, {}, 1 ), 0 ); // everything is Delone now
}

} //namespace MR
