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

TEST(MRMesh, QuadraticForm)
{
    QuadraticForm3f q0, q1;
    q0.addDistToOrigin( 1 );
    q1.addDistToOrigin( 1 );
    auto r = sum( q0, Vector3f{0,0,0}, q1, Vector3f{2,0,0} );

    EXPECT_EQ( r.second, (Vector3f{1,0,0}) );
}

} //namespace MR
