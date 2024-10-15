#include "MRBox.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRGTest.h"

namespace MR
{

// verifies that template can be instantiated with typical parameters
template struct Box<float>;
template struct Box<double>;
template struct Box<Vector2f>;
template struct Box<Vector2d>;
template struct Box<Vector3f>;
template struct Box<Vector3d>;

TEST(MRMesh, Box) 
{
    Box3d b1{ Vector3d{ 0, 0, 0 }, Vector3d{ 1, 1, 1 } };
    Box3d b2{ Vector3d{ -1, -1, -1 }, Vector3d{ 1, 1, 1 } };
    Box3d b3{ Vector3d{ 2, 2, 2 }, Vector3d{ 3, 3, 3 } };

    EXPECT_TRUE( b1.intersects( b2 ) );
    EXPECT_EQ( b1.intersection( b2 ), b1 );
    EXPECT_EQ( Box3d{ b1 }.intersect( b2 ), b1 );

    EXPECT_FALSE( b1.intersects( b3 ) );
    EXPECT_FALSE( b1.intersection( b3 ).valid() );

    Box3i b = { {0, 0, 0}, {10, 10, 10} };
    Vector3i c = b.center();
    Vector3i r{ 5, 5, 5 };
    EXPECT_EQ( c, r );
}

} //namespace MR
