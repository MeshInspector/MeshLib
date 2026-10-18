#include <MRMesh/MRQuadraticForm.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRVector3.h>
#include <gtest/gtest.h>

namespace MR
{

// verifies that template can be instantiated with typical parameters
template struct QuadraticForm<Vector2<float>>;
template struct QuadraticForm<Vector2<double>>;
template struct QuadraticForm<Vector3<float>>;
template struct QuadraticForm<Vector3<double>>;

// the minimum of the sum of well-conditioned forms is found by direct inversion, and matches pseudoinverse;
// a degenerate sum (two parallel planes) falls back on pseudoinverse
TEST( MRMesh, QuadraticFormSum )
{
    const Vector3d x0( 1, 2, 3 ), x1( 2, 0, 1 );
    QuadraticForm3d q0, q1;
    q0.addDistToPlane( Vector3d( 1, 0, 0 ) );
    q0.addDistToPlane( Vector3d( 0, 1, 1 ).normalized() );
    q0.addDistToOrigin( 0.001 );
    q1.addDistToPlane( Vector3d( 0, 0, 1 ) );
    q1.addDistToLine( Vector3d( 1, 1, 0 ).normalized() );

    auto pinvMin = []( const QuadraticForm3d & a, const Vector3d & y0, const QuadraticForm3d & b, const Vector3d & y1 )
    {
        const auto center = 0.5 * ( y0 + y1 );
        return ( a.A + b.A ).pseudoinverse() * ( a.A * ( y0 - center ) + b.A * ( y1 - center ) ) + center;
    };

    const auto [q2, x2] = sum( q0, x0, q1, x1 );
    const auto ref = pinvMin( q0, x0, q1, x1 );
    EXPECT_NEAR( ( x2 - ref ).length(), 0, 1e-9 );
    EXPECT_NEAR( q2.c, q0.eval( x0 - ref ) + q1.eval( x1 - ref ), 1e-9 );

    QuadraticForm3d p0, p1;
    p0.addDistToPlane( Vector3d( 0, 0, 1 ) );
    p1.addDistToPlane( Vector3d( 0, 0, 1 ), 2 );
    const auto [p2, y2] = sum( p0, x0, p1, x1 );
    EXPECT_EQ( y2, pinvMin( p0, x0, p1, x1 ) );

    // the same in 2D and in float
    QuadraticForm2f r0, r1;
    r0.addDistToPlane( Vector2f( 1, 0 ) );
    r1.addDistToPlane( Vector2f( 1, 1 ).normalized() );
    const auto [r2, z2] = sum( r0, Vector2f( 1, 2 ), r1, Vector2f( 3, 1 ) );
    EXPECT_NEAR( ( z2 - Vector2f( 1, 3 ) ).length(), 0, 1e-5f ); // x = 1 and x + y = 4
}

} //namespace MR
