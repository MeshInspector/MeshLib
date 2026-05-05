#include <MRMesh/MRReducePath.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, ShortestPathInQuadrangleTest )
{
    Vector3d a;
    Vector3d b = Vector3d( 0, 1, 0 );
    Vector3d c = Vector3d( 1, 1, 0 );
    Vector3d d = Vector3d( 1, 0, 0 );
    // common case
    auto x = shortestPathInQuadrangle( a, b, c, d );
    EXPECT_NEAR( x, 0.5, 1e-15 );

    c.x = c.y = 0.1;
    // concave case
    x = shortestPathInQuadrangle( a, b, c, d );
    EXPECT_TRUE( x == 1 );

    b = Vector3d( 0, 1, 0 );
    a = c = Vector3d( 0.5, 0.5, 0 );
    // degenerate needles case
    x = shortestPathInQuadrangle( a, b, c, d );
    EXPECT_TRUE( x == 0 );

    a = Vector3d();
    c = Vector3d( 1, 1, 0 );
    b = d = Vector3d( 0.5, 0.5, 0 );
    // degenerate caps case
    x = shortestPathInQuadrangle( a, b, c, d );
    EXPECT_NEAR( x, 0.5, 1e-15 );
}

}
