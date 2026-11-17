#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRMatrix3Decompose.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRQuaternion.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MRLine3.h>
#include <gtest/gtest.h>

namespace MR
{

TEST(MRMesh, AffineXf3)
{
    ASSERT_EQ( Vector3f::plusX() + Vector3f::minusX(), Vector3f() );
    ASSERT_EQ( Vector3f::plusY() + Vector3f::minusY(), Vector3f() );
    ASSERT_EQ( Vector3f::plusZ() + Vector3f::minusZ(), Vector3f() );

    Vector3d pt{ 1., 1., 1. };
    ASSERT_EQ( Matrix3d::scale( 1.5 ) * pt, ( Vector3d{ 1.5, 1.5, 1.5 } ) );

    auto shiftXf = AffineXf3d::translation( { 1., 0., 0. } );
    ASSERT_EQ( shiftXf( pt ), ( Vector3d{ 2., 1., 1. } ) );

    auto rot = Matrix3d::rotation( Vector3d{ 1., 1., 1. }, PI * 2 / 3 );
    ASSERT_NEAR( ( rot * Vector3d{ 1., 0., 0. } - Vector3d{ 0., 1., 0. } ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( rot * Vector3d{ 0., 1., 0. } - Vector3d{ 0., 0., 1. } ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( rot * Vector3d{ 0., 0., 1. } - Vector3d{ 1., 0., 0. } ).length(), 0., 1e-15 );

    auto rotXf = AffineXf3d::linear( rot );
    // test composition
    auto u = rotXf( shiftXf( pt ) );
    auto xf = rotXf * shiftXf;
    auto v = xf( pt );
    ASSERT_NEAR( ( u - v ).length(), 0., 1e-15 );
    auto invXf = xf.inverse();
    ASSERT_NEAR( ( invXf( v ) - pt ).length(), 0., 1e-15 );

    Vector3d p0{ 1., 0., 0. };
    Vector3d p1{ 0., 1., 0. };
    auto rotXY = Matrix3d::rotation( p0, p1 );
    ASSERT_NEAR( ( rotXY * p0 - p1 ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( rotXY.inverse() * p1 - p0 ).length(), 0., 1e-15 );
    auto rotXX = Matrix3d::rotation( p0, p0 );
    ASSERT_NEAR( ( rotXX * p0 - p0 ).length(), 0., 1e-15 );
    auto rot_XX = Matrix3d::rotation( p0, -p0 );
    ASSERT_NEAR( ( rot_XX * p0 + p0 ).length(), 0., 1e-15 );

    const auto m0 = Matrix3d::rotation( Vector3d::plusZ(), PI / 2 );
    auto m1 = Matrix3d::identity();
    ASSERT_NEAR( ( Quaterniond::slerp( m0, m1, 0.0 ) - m0 ).norm(), 0., 1e-15 );
    ASSERT_NEAR( ( Quaterniond::slerp( m0, m1, 0.5 ) - Matrix3d::rotation( Vector3d::plusZ(), PI / 4 ) ).norm(), 0., 1e-15 );
    ASSERT_NEAR( ( Quaterniond::slerp( m0, m1, 1.0 ) - m1 ).norm(), 0., 1e-15 );

    const auto b0 = Vector3d{0,0,0};
    const auto xf0 = AffineXf3d{ m0, b0 };
    const auto b1 = Vector3d{1,1,1};
    const auto xf1 = AffineXf3d{ m1, b1 };
    ASSERT_NEAR( ( Quaterniond::slerp( xf0, xf1, 0.0 )( Vector3d(0,0,0) ) - b0 ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( Quaterniond::slerp( xf0, xf1, 0.5 )( Vector3d(0,0,0) ) - Vector3d(0.5,0.5,0.5) ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( Quaterniond::slerp( xf0, xf1, 1.0 )( Vector3d(0,0,0) ) - b1 ).length(), 0., 1e-15 );

    auto center = Vector3d{1,-1,0};
    ASSERT_NEAR( ( Quaterniond::slerp( xf0, xf1, 0.0, center )( Vector3d(0,0,0) ) - b0 ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( Quaterniond::slerp( xf0, xf1, 0.5, center )( center ) - 0.5 * ( xf0( center ) + xf1( center ) ) ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( Quaterniond::slerp( xf0, xf1, 1.0, center )( Vector3d(0,0,0) ) - b1 ).length(), 0., 1e-15 );

    const auto xf2 = AffineXf3d{ m0, b1 };
    Plane3d pl{ Vector3d{1,1,-1}.normalized(), 1.0 };
    Plane3d xPl = transformed( pl, xf2 );
    ASSERT_NEAR( pl.distance( Vector3d{1,0,-1} ) - xPl.distance( xf2( Vector3d{1,0,-1} ) ), 0., 1e-15 );
    Plane3d plA = invTransformed( xPl, xf2 );
    ASSERT_NEAR( ( pl.n - plA.n ).length(), 0., 1e-15 );
    ASSERT_NEAR( pl.d - plA.d, 0., 1e-15 );

    Line3d line{ Vector3d{-1,-1,1}, Vector3d{1,1,-1}.normalized() };
    Line3d xLine = transformed( line, xf2 );
    ASSERT_NEAR( line.distanceSq( Vector3d{1,0,-1} ) - xLine.distanceSq( xf2( Vector3d{1,0,-1} ) ), 0., 1e-15 );
    Line3d lineA = transformed( xLine, xf2.inverse() );
    ASSERT_NEAR( ( line.p - lineA.p ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( line.d - lineA.d ).length(), 0., 1e-15 );

    for ( double d : Vector3d{1,1,1} )
        ASSERT_EQ( d, 1 );

    // this failed with "Alternative Method" inside Quaternion<T>::Quaternion( const Matrix3<T> & m )
    Matrix3d m;
    m.x = Vector3d{ -0.942630529,  0.230237782,  0.241739601 };
    m.y = Vector3d{  0.230237812, -0.0759973153, 0.970162332 };
    m.z = Vector3d{  0.241739616,  0.970162332,  0.0186279733 };
    ASSERT_NEAR( ( orthonormalized( m ) - m ).norm(), 0, 1e-6 );

    Matrix3d rr( { 1., 2., 3. }, { 0., 4., 5. }, { 0., 0., 6. } );
    const auto [q, r] = rr.qr();
    ASSERT_NEAR( ( q - Matrix3d::identity() ).norm(), 0, 1e-12 );
    ASSERT_NEAR( ( r - rr ).norm(), 0, 1e-12 );
}

TEST(MRMesh, DecomposeMatrix3)
{
    const auto rot = Matrix3f::rotationFromEuler( { 0.3f, -0.7f, 1.9f } );
    const Matrix3f mirrorX = Matrix3f::scale( -1.f, 1.f, 1.f ), mirrorY = Matrix3f::scale( 1.f, -1.f, 1.f ), mirrorZ = Matrix3f::scale( 1.f, 1.f, -1.f );
    for ( const Matrix3f& m : { Matrix3f{}, rot, mirrorX, mirrorY, mirrorZ, Matrix3f::scale( -1.f ), rot * Matrix3f::scale( 2.f, -0.5f, 3.f ) } )
    {
        for ( bool signedScale : { false, true } )
        {
            Matrix3f r, s;
            decomposeMatrix3( m, r, s, signedScale );
            const bool mirror = m.det() < 0;
            EXPECT_NEAR( ( r * s - m ).norm(), 0, 1e-5f );
            EXPECT_NEAR( r.det(), ( !mirror || signedScale ) ? 1 : -1, 1e-5f );
            EXPECT_EQ( s.x.x * s.y.y * s.z.z < 0, mirror && signedScale );
            if ( signedScale )
            {
                // the round trip of MCP and the Scene-info panel: rotation to Euler angles and back
                EXPECT_NEAR( ( Matrix3f::rotationFromEuler( r.toEulerAngles() ) * s - m ).norm(), 0, 1e-5f );
            }
        }
    }

    // axis mirrors read back as written (with the identity rotation), and a singular matrix is not a mirror
    for ( const Matrix3f& m : { mirrorX, mirrorY, mirrorZ, Matrix3f::scale( 1.f, 1.f, 0.f ) } )
    {
        Matrix3f r, s;
        decomposeMatrix3( m, r, s, true );
        EXPECT_EQ( s, m );
    }
}

// verifies that template can be instantiated with typical parameters
template struct AffineXf<Vector3<float>>;
template struct AffineXf<Vector3<double>>;

} //namespace MR
