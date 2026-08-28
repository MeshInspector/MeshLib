#include <MRMesh/MRRayBoxIntersection.h>
#include <gtest/gtest.h>
#include <cfloat>
#include <cmath>
#include <random>

namespace MR
{

namespace
{

Vector3f invertDir( const Vector3f & dir )
{
    Vector3f res;
    for ( int i = 0; i < 3; ++i )
        res[i] = ( dir[i] == 0 ) ? std::numeric_limits<float>::max() : 1 / dir[i];
    return res;
}

/// straightforward implementation to validate the hardware optimized one against
bool refRayBoxIntersect( const Box3f & box, const Vector3f & org, float & t0, float & t1, const Vector3f & invDir )
{
    for ( int i = 0; i < 3; ++i )
    {
        const float a = ( box.min[i] - org[i] ) * invDir[i];
        const float b = ( box.max[i] - org[i] ) * invDir[i];
        t0 = std::max( std::min( a, b ), t0 );
        t1 = std::min( std::max( a, b ), t1 );
    }
    return t0 <= t1;
}

} //anonymous namespace

TEST( MRMesh, RayBoxIntersect )
{
    const Box3f box{ Vector3f{ 1, 1, 1 }, Vector3f{ 2, 2, 2 } };
    const Vector3f org{ 0, 1.5f, 1.5f };
    const IntersectionPrecomputes<float> alongX( Vector3f{ 1, 0, 0 } );

    float t0 = 0, t1 = FLT_MAX;
    EXPECT_TRUE( rayBoxIntersect( box, RayOrigin<float>{ org }, t0, t1, alongX ) );
    EXPECT_EQ( t0, 1 );
    EXPECT_EQ( t1, 2 );

    // the box is beyond the segment end
    t0 = 0; t1 = 0.5f;
    EXPECT_FALSE( rayBoxIntersect( box, RayOrigin<float>{ org }, t0, t1, alongX ) );

    // the ray is directed away from the box
    t0 = 0; t1 = FLT_MAX;
    EXPECT_FALSE( rayBoxIntersect( box, RayOrigin<float>{ org }, t0, t1,
        IntersectionPrecomputes<float>( Vector3f{ -1, 0, 0 } ) ) );

    // the ray is parallel to the box and misses it
    t0 = 0; t1 = FLT_MAX;
    EXPECT_FALSE( rayBoxIntersect( box, RayOrigin<float>{ Vector3f{ 0, 0, 1.5f } }, t0, t1, alongX ) );
}

// rayBoxIntersect for floats is implemented in SIMD instructions on some platforms,
// here it is verified to produce exactly the same results as the straightforward code
TEST( MRMesh, RayBoxIntersectSimd )
{
    std::mt19937 gen( 12345 );
    std::uniform_real_distribution<float> coord( -10, 10 );
    // small integer directions have zero components, which turn into huge inverted directions
    std::uniform_int_distribution<int> smallInt( -2, 2 );

    for ( int i = 0; i < 10000; ++i )
    {
        Box3f box;
        box.include( Vector3f{ coord( gen ), coord( gen ), coord( gen ) } );
        box.include( Vector3f{ coord( gen ), coord( gen ), coord( gen ) } );

        const Vector3f org{ coord( gen ), coord( gen ), coord( gen ) };
        const Vector3f dir = ( i % 4 == 0 )
            ? Vector3f{ float( smallInt( gen ) ), float( smallInt( gen ) ), float( smallInt( gen ) ) }
            : Vector3f{ coord( gen ), coord( gen ), coord( gen ) };
        if ( dir == Vector3f{} )
            continue;

        const float rayEnd = ( i % 3 == 0 ) ? std::abs( coord( gen ) ) : FLT_MAX;

        float t0 = 0, t1 = rayEnd;
        const bool res = rayBoxIntersect( box, RayOrigin<float>{ org }, t0, t1, IntersectionPrecomputes<float>( dir ) );

        float refT0 = 0, refT1 = rayEnd;
        const bool refRes = refRayBoxIntersect( box, org, refT0, refT1, invertDir( dir ) );

        ASSERT_EQ( res, refRes ) << "iteration " << i;
        ASSERT_EQ( t0, refT0 ) << "iteration " << i;
        ASSERT_EQ( t1, refT1 ) << "iteration " << i;
    }
}

} //namespace MR
