#include <MRMesh/MRBox.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRGTest.h>

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

    EXPECT_TRUE( b1.contains( b1 ) );
    EXPECT_TRUE( b2.contains( b1 ) );
    EXPECT_FALSE( b1.contains( b2 ) );

    Box3i b = { {0, 0, 0}, {10, 10, 10} };
    Vector3i c = b.center();
    Vector3i r{ 5, 5, 5 };
    EXPECT_EQ( c, r );

    auto p11 = getTouchPlanes( b1, Vector3d( 1, 0, 0 ) );
    EXPECT_EQ( p11.min, 0 );
    EXPECT_EQ( p11.max, 1 );
    auto p12 = getTouchPlanes( b1, Vector3d( 1, 1, 1 ) );
    EXPECT_EQ( p12.min, 0 );
    EXPECT_EQ( p12.max, 3 );

    Box3d b4{ Vector3d{ 1, 2, 3 }, Vector3d{ 4, 5, 6 } };
    EXPECT_EQ( b4.corner( Vector3b( 1, 0, 1 ) ), Vector3d( 4, 2, 6 ) );
    EXPECT_EQ( b4.getMinBoxCorner( Vector3d( -1, 1, -2 ) ), Vector3b( 1, 0, 1 ) );

    MinMaxf mm0{ -1, 2 };
    EXPECT_EQ( mm0.corner( 0 ), -1 );
    EXPECT_EQ( mm0.corner( 1 ),  2 );
    EXPECT_EQ( mm0.getMinBoxCorner(  1 ), 0 );
    EXPECT_EQ( mm0.getMinBoxCorner( -1 ), 1 );

    EXPECT_EQ( findSortedBoxDims( Box3f( { 0, 0, 0 }, { 2, 1, 3 } ) ), Vector3i( 1, 0, 2 ) );
}

} //namespace MR
