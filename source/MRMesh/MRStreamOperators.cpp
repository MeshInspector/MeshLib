#include "MRStreamOperators.h"
#include "MRBox.h"
#include "MRGTest.h"
#include <cassert>

namespace MR
{
TEST( MRMesh, StreamOperators )
{
    {
        std::stringstream ss;
        Vector2f v1( 1.1f, -0.3f );
        Vector2f v2;
        ss << v1;
        ss >> v2;
        EXPECT_TRUE( v1 == v2 );
    }

    {
        std::stringstream ss;
        Vector3f v1( 1.0f, -2.0f, -0.3f );
        Vector3f v2;
        ss << v1;
        ss >> v2;
        EXPECT_TRUE( v1 == v2 );
    }

    {
        std::stringstream ss;
        Vector4f v1( 1.0f, -2.0f, 0.3f, -0.4f );
        Vector4f v2;
        ss << v1;
        ss >> v2;
        EXPECT_TRUE( v1 == v2 );
    }

    {
        std::stringstream ss;
        Vector3f v1( 1.0f, 2.0f, 3.0f );
        Vector3f v2( -4.0f, -5.0f, -6.0f );
        Vector3f v3( 0.7f, -0.8f, 0.9f );
        Matrix3<float> m1( v1, v2, v3 );
        Matrix3<float> m2;
        ss << m1;
        ss >> m2;
        EXPECT_TRUE( m1 == m2 );
    }

    {
        std::stringstream ss;
        Vector3f v1( 1.0f, 2.0f, 3.0f );
        Plane3f p1( v1, -5.2f );
        Plane3f p2;
        ss << p1;
        ss >> p2;
        EXPECT_TRUE( p1 == p2 );
    }

    {
        std::stringstream ss;
        TriPoint<float> tp1( 0.8f, 0.1f );
        TriPoint<float> tp2;
        ss << tp1;
        ss >> tp2;
        EXPECT_TRUE( tp1.a == tp2.a );
        EXPECT_TRUE( tp1.b == tp2.b );
    }

    {
        std::stringstream ss;
        Vector3f v1( 1.0f, 2.0f, 3.0f );
        Vector3f v2( -4.0f, -5.0f, -6.0f );
        Vector3f v3( 0.7f, -0.8f, 0.9f );
        Vector3f v4( -1.0f, -1.1f, 1.2f );
        Matrix3<float> m1( v1, v2, v3 );
        AffineXf3<float> xf1( m1, v4 );
        AffineXf3<float> xf2;
        ss << xf1;
        ss >> xf2;
        EXPECT_TRUE( xf1 == xf2 );
    }

    {
        std::stringstream ss;
        Vector3f v1( 1.0f, 2.0f, 3.0f );
        PointOnFace pof1;
        pof1.face = FaceId( 10 );
        pof1.point = v1;
        PointOnFace pof2;
        ss << pof1;
        ss >> pof2;
        EXPECT_TRUE( pof1.face == pof2.face );
        EXPECT_TRUE( pof1.point == pof2.point );
    }

    {
        std::stringstream ss;
        Box3f b1( { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } );
        Box3f b2;
        ss << b1;
        ss >> b2;
        EXPECT_TRUE( b1 == b2 );
    }
}

}
