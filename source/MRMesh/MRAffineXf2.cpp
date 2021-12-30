#include "MRAffineXf2.h"
#include "MRConstants.h"
#include "MRGTest.h"

namespace MR
{

// verifies that template can be instantiated with typical parameters
template struct AffineXf<Vector2<float>>;
template struct AffineXf<Vector2<double>>;

TEST(MRMesh, AffineXf2) 
{
    ASSERT_EQ( Vector2f::plusX() + Vector2f::minusX(), Vector2f() );
    ASSERT_EQ( Vector2f::plusY() + Vector2f::minusY(), Vector2f() );

    Vector2d pt{ 1., 1. };
    ASSERT_EQ( Matrix2d::scale( 1.5 ) * pt, ( Vector2d{ 1.5, 1.5 } ) );

    auto shiftXf = AffineXf2d::translation( { 1., 0. } );
    ASSERT_EQ( shiftXf( pt ), ( Vector2d{ 2., 1. } ) );

    auto rot = Matrix2d::rotation( PI / 2 );
    ASSERT_NEAR( ( rot * Vector2d{ 1., 0. } - Vector2d{ 0.,  1. } ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( rot * Vector2d{ 0., 1. } - Vector2d{ -1., 0. } ).length(), 0., 1e-15 );

    auto rotXf = AffineXf2d::linear( rot );
    // test composition
    auto u = rotXf( shiftXf( pt ) );
    auto xf = rotXf * shiftXf;
    auto v = xf( pt );
    ASSERT_NEAR( ( u - v ).length(), 0., 1e-15 );
    auto invXf = xf.inverse();
    ASSERT_NEAR( ( invXf( v ) - pt ).length(), 0., 1e-15 );

    Vector2d p0{ 1., 0. };
    Vector2d p1{ 0., 1. };
    auto rotXY = Matrix2d::rotation( p0, p1 );
    ASSERT_NEAR( ( rotXY * p0 - p1 ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( rotXY.inverse() * p1 - p0 ).length(), 0., 1e-15 );
    auto rotXX = Matrix2d::rotation( p0, p0 );
    ASSERT_NEAR( ( rotXX * p0 - p0 ).length(), 0., 1e-15 );
    auto rot_XX = Matrix2d::rotation( p0, -p0 );
    ASSERT_NEAR( ( rot_XX * p0 + p0 ).length(), 0., 1e-15 );

    for ( double d : Vector2d{1,1} )
        ASSERT_EQ( d, 1 );
}

} //namespace MR
