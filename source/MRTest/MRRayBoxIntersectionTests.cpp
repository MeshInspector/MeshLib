#include <MRMesh/MRLine3.h>
#include <MRMesh/MRRayBoxIntersection.h>
#include <gtest/gtest.h>
#include <cfloat>
#include <limits>

namespace MR
{

namespace
{

// the cases are given in doubles and checked in both float and double, so that both the SIMD
// specializations and the generic implementation are verified against the same expected values;
// every value is exactly representable in float, so the comparisons below are exact
struct RayBoxCase
{
    const char * name;
    Vector3d boxMin, boxMax;
    Vector3d org, dir;
    double rayStart, rayEnd; // cUnbounded means the largest representable value
    bool expected;
    double t0, t1;           // only checked when the ray does intersect
};

constexpr double cUnbounded = -1;

const RayBoxCase cRayBoxCases[] = {
    { "hit along +x",
      { 1, 1, 1 }, { 2, 2, 2 }, { 0, 1.5, 1.5 }, { 1, 0, 0 }, 0, cUnbounded, true, 1, 2 },
    { "oblique hit",
      { 1, 1, 1 }, { 2, 2, 2 }, { 0, 0.5, 1.5 }, { 2, 1, 0 }, 0, cUnbounded, true, 0.5, 1 },
    { "hit from the far side, negative direction",
      { 1, 1, 1 }, { 2, 2, 2 }, { 3, 1.5, 1.5 }, { -1, 0, 0 }, 0, cUnbounded, true, 1, 2 },
    { "diagonal through the opposite corners",
      { 1, 1, 1 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, 1 }, 0, cUnbounded, true, 1, 2 },
    { "origin inside the box",
      { 1, 1, 1 }, { 2, 2, 2 }, { 1.5, 1.5, 1.5 }, { 1, 0, 0 }, 0, cUnbounded, true, 0, 0.5 },
    { "origin exactly on the near face",
      { 1, 1, 1 }, { 2, 2, 2 }, { 1, 1.5, 1.5 }, { 1, 0, 0 }, 0, cUnbounded, true, 0, 1 },
    // the zero direction components make the box bounds coincide with the origin along y and z,
    // which is the reason the inverted direction is limited to the largest value instead of infinity
    { "origin exactly on the near corner",
      { 1, 1, 1 }, { 2, 2, 2 }, { 1, 1, 1 }, { 1, 0, 0 }, 0, cUnbounded, true, 0, 1 },
    { "grazing the box edge",
      { 1, 1, 1 }, { 2, 2, 2 }, { 0, 1, 1 }, { 1, 0, 0 }, 0, cUnbounded, true, 1, 2 },
    { "flat box, ray along its normal",
      { 1, 1, 1 }, { 2, 2, 1 }, { 1.5, 1.5, 0 }, { 0, 0, 1 }, 0, cUnbounded, true, 1, 1 },
    { "segment starting inside the box",
      { 1, 1, 1 }, { 2, 2, 2 }, { 0, 1.5, 1.5 }, { 1, 0, 0 }, 1.5, cUnbounded, true, 1.5, 2 },
    { "directed away from the box",
      { 1, 1, 1 }, { 2, 2, 2 }, { 0, 1.5, 1.5 }, { -1, 0, 0 }, 0, cUnbounded, false, 0, 0 },
    { "box beyond the segment end",
      { 1, 1, 1 }, { 2, 2, 2 }, { 0, 1.5, 1.5 }, { 1, 0, 0 }, 0, 0.5, false, 0, 0 },
    { "parallel to the box and missing it",
      { 1, 1, 1 }, { 2, 2, 2 }, { 0, 0, 1.5 }, { 1, 0, 0 }, 0, cUnbounded, false, 0, 0 },
    { "diagonal missing the box",
      { 1, 1, 1 }, { 2, 2, 2 }, { 0, 0, 0 }, { 1, 1, -1 }, 0, cUnbounded, false, 0, 0 },
};

template <typename T>
void testRayBoxCases()
{
    for ( const auto & c : cRayBoxCases )
    {
        const Box3<T> box{ Vector3<T>( c.boxMin ), Vector3<T>( c.boxMax ) };
        T t0 = T( c.rayStart );
        T t1 = c.rayEnd == cUnbounded ? std::numeric_limits<T>::max() : T( c.rayEnd );
        const bool res = rayBoxIntersect( box, RayOrigin<T>( Vector3<T>( c.org ) ), t0, t1,
            IntersectionPrecomputes<T>( Vector3<T>( c.dir ) ) );
        EXPECT_EQ( res, c.expected ) << c.name;
        if ( !c.expected || !res )
            continue;
        EXPECT_EQ( t0, T( c.t0 ) ) << c.name;
        EXPECT_EQ( t1, T( c.t1 ) ) << c.name;
    }
}

} //anonymous namespace

TEST( MRMesh, RayBoxIntersect )
{
    // float takes the SIMD specialization where the platform has one, double is always generic
    testRayBoxCases<float>();
    testRayBoxCases<double>();

    // the overload taking a line and computing the precomputes itself
    const Box3f box{ Vector3f{ 1, 1, 1 }, Vector3f{ 2, 2, 2 } };
    EXPECT_TRUE( rayBoxIntersect( box, Line3f{ Vector3f{ 0, 1.5f, 1.5f }, Vector3f{ 1, 0, 0 } }, 0.f, FLT_MAX ) );
    EXPECT_FALSE( rayBoxIntersect( box, Line3f{ Vector3f{ 0, 1.5f, 1.5f }, Vector3f{ 1, 0, 0 } }, 0.f, 0.5f ) );
}

} //namespace MR
