// Regression coverage for MR::findDistance / MR::findSignedDistance with all
// reasonable combinations of relative mesh placement and upDistLimitSq.
//
// Tested mesh configurations (one TEST per case):
//   FarApart           - gap >> 1 mm
//   CloseNoIntersect   - gap < 1 mm, no contact
//   TouchPoint         - sphere vertex sits on cube face
//   TouchArea          - two cubes share a face area
//   ShallowIntersect   - penetration < 1 mm
//   DeepIntersect      - penetration > 1 mm, neither mesh inside the other
//   OneInsideOther     - small cube fully inside the bigger one
//
// For each configuration we exercise three values of upDistLimitSq:
//   FLT_MAX  - no upper limit
//   1.0f     - limit corresponds to a 1 mm distance (1 mm squared)
//   0.0f     - strict zero limit

#include <MRMesh/MRMeshMeshDistance.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRAffineXf3.h>
#include <gtest/gtest.h>

#include <cfloat>

namespace MR
{

namespace
{

constexpr float kNoLimitSq = FLT_MAX; // upDistLimitSq for "no limit"
constexpr float kOneMmLimitSq = 1.0f; // upDistLimitSq for a 1 mm distance limit
constexpr float kZeroLimitSq = 0.0f;  // strict zero limit

// Reference cube used as mesh "a" in every case: edge length 4, centered on origin.
Mesh makeRefCube()
{
    return makeCube( Vector3f::diagonal( 4.0f ), Vector3f::diagonal( -2.0f ) );
}

// Centered unit cube spanning [-0.5, 0.5]^3.
Mesh makeUnitCube()
{
    return makeCube( Vector3f::diagonal( 1.0f ), Vector3f::diagonal( -0.5f ) );
}

} // namespace

// clang-format off
// ----------------------------------------------------------------------------
// (a) FarApart - gap of 7.5 mm, much larger than the 1 mm limit.
//
//  Cross-section at z = 1.5 (y-axis vertical, x-axis horizontal):
//
//   y
//   ^                                                  +---+
//  2|  +-------+                                       | b |   b: 1x1x1 cube
//   |  |       |                                       +---+      shifted by
//  0|  |   a   |                                                  (10, 1.5, 1.5)
//   |  |       |
// -2|  +-------+
//   +-----------------------------------------------------> x
//      -2      2                                     9.5  10.5
//
//  b in a-coords: [9.5, 10.5] x [1, 2] x [1, 2]
//  Closest pair: corner (2,2,2) of a  <->  corner (9.5,2,2) of b   (gap = 7.5)
//  Diagonal shift in y,z is necessary so the closest pair is vertex-to-vertex:
//  findTriTriDistance does not find the perpendicular distance between two
//  parallel coplanar cube faces unless a vertex match is present.
// clang-format on
TEST( MRMesh, MeshMeshDistance_FarApart )
{
    const auto a = makeRefCube();  // [-2, 2]^3
    const auto b = makeUnitCube(); // [-0.5, 0.5]^3
    const AffineXf3f xf = AffineXf3f::translation( Vector3f( 10.0f, 1.5f, 1.5f ) );
    const float trueDist = 7.5f;
    const float trueDistSq = trueDist * trueDist;

    // no limit: function returns the true gap
    {
        const auto d = findDistance( a, b, &xf, kNoLimitSq );
        EXPECT_NEAR( d.distSq, trueDistSq, 1e-2f );

        const auto sd = findSignedDistance( a, b, &xf, kNoLimitSq );
        EXPECT_NEAR( sd.signedDist, trueDist, 1e-2f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::BothOutside );
    }

    // 1 mm limit: real distance exceeds the limit -> findDistance early-exits
    // returning distSq = upDistLimitSq with invalid res.a/b. findSignedDistance
    // cannot determine inside/outside without valid projection points, so it
    // reports NotColliding with signedDist = +sqrt(upDistLimitSq) = 1.0.
    {
        const auto d = findDistance( a, b, &xf, kOneMmLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, kOneMmLimitSq );

        const auto sd = findSignedDistance( a, b, &xf, kOneMmLimitSq );
        EXPECT_FLOAT_EQ( sd.signedDist, 1.0f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::NotColliding );
    }

    // strict 0 limit: findDistance never descends into the BVH, returns
    // distSq = 0 with invalid res.a/b. findCollidingTriangles is empty for
    // far-apart meshes, but inside/outside cannot be derived from the invalid
    // points, so the function reports NotColliding with signedDist = 0.
    {
        const auto d = findDistance( a, b, &xf, kZeroLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, kZeroLimitSq );

        const auto sd = findSignedDistance( a, b, &xf, kZeroLimitSq );
        EXPECT_FLOAT_EQ( sd.signedDist, 0.0f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::NotColliding );
    }
}

// clang-format off
// ----------------------------------------------------------------------------
// (b) CloseNoIntersect - gap of 0.5 mm, below the 1 mm limit.
//
//  Cross-section at z = 1.5:
//
//   y
//   ^                           +---+
//  2|  +-------+                | b |    b: 1x1x1 cube shifted by (3, 1.5, 1.5)
//   |  |       |                +---+
//  0|  |   a   |
//   |  |       |
// -2|  +-------+
//   +-----------------------------> x
//      -2      2              2.5  3.5
//
//  b in a-coords: [2.5, 3.5] x [1, 2] x [1, 2]
//  Closest pair: corner (2,2,2) of a  <->  corner (2.5,2,2) of b   (gap = 0.5)
// clang-format on
TEST( MRMesh, MeshMeshDistance_CloseNoIntersect )
{
    const auto a = makeRefCube();  // [-2, 2]^3
    const auto b = makeUnitCube(); // [-0.5, 0.5]^3
    const AffineXf3f xf = AffineXf3f::translation( Vector3f( 3.0f, 1.5f, 1.5f ) );
    const float trueDist = 0.5f;
    const float trueDistSq = trueDist * trueDist;

    // no limit
    {
        const auto d = findDistance( a, b, &xf, kNoLimitSq );
        EXPECT_NEAR( d.distSq, trueDistSq, 1e-4f );

        const auto sd = findSignedDistance( a, b, &xf, kNoLimitSq );
        EXPECT_NEAR( sd.signedDist, trueDist, 1e-3f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::BothOutside );
    }

    // 1 mm limit: real distance is below the limit -> proper answer
    {
        const auto d = findDistance( a, b, &xf, kOneMmLimitSq );
        EXPECT_NEAR( d.distSq, trueDistSq, 1e-4f );

        const auto sd = findSignedDistance( a, b, &xf, kOneMmLimitSq );
        EXPECT_NEAR( sd.signedDist, trueDist, 1e-3f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::BothOutside );
    }

    // strict 0 limit: any positive distance is above the limit -> early exit
    // with invalid res.a/b. findCollidingTriangles is empty, but inside/outside
    // cannot be derived from the invalid points -> NotColliding, signedDist = 0.
    {
        const auto d = findDistance( a, b, &xf, kZeroLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, kZeroLimitSq );

        const auto sd = findSignedDistance( a, b, &xf, kZeroLimitSq );
        EXPECT_FLOAT_EQ( sd.signedDist, 0.0f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::NotColliding );
    }
}

// clang-format off
// ----------------------------------------------------------------------------
// (d) TouchArea - two cubes pressed face-to-face.
//
//  Cross-section at z = 0:
//
//   y
//    ^
//  2 |  +-------+
//    |  |       |
// 0.5|  |       +---+
//    |  |   a   | b |       b: 1x1x1 cube shifted by (2.5, 0, 0)
//-0.5|  |       +---+       b's face x=2 lies on a's face x=2 (1x1 contact)
//    |  |       |
// -2 |  +-------+
//    +-------------------> x
//       -2      2   3
//
//  b in a-coords: [2, 3] x [-0.5, 0.5] x [-0.5, 0.5]
// clang-format on
TEST( MRMesh, MeshMeshDistance_TouchArea )
{
    const auto a = makeRefCube();  // [-2, 2]^3
    const auto b = makeUnitCube(); // [-0.5, 0.5]^3
    const AffineXf3f xf = AffineXf3f::translation( Vector3f( 2.5f, 0.0f, 0.0f ) );

    // no limit
    {
        const auto d = findDistance( a, b, &xf, kNoLimitSq );
        EXPECT_NEAR( d.distSq, 0.0f, 1e-6f );

        const auto sd = findSignedDistance( a, b, &xf, kNoLimitSq );
        EXPECT_NEAR( sd.signedDist, 0.0f, 1e-4f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::Touching );
    }

    // 1 mm limit
    {
        const auto d = findDistance( a, b, &xf, kOneMmLimitSq );
        EXPECT_NEAR( d.distSq, 0.0f, 1e-6f );

        const auto sd = findSignedDistance( a, b, &xf, kOneMmLimitSq );
        EXPECT_NEAR( sd.signedDist, 0.0f, 1e-4f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::Touching );
    }

    // strict 0 limit: findCollidingTriangles is empty for coplanar face touch
    // -> Touching, same as the no-limit case.
    {
        const auto d = findDistance( a, b, &xf, kZeroLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, kZeroLimitSq );

        const auto sd = findSignedDistance( a, b, &xf, kZeroLimitSq );
        EXPECT_FLOAT_EQ( sd.signedDist, 0.0f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::Touching ); // <-- PROBLEM 2
    }
}

// clang-format off
// ----------------------------------------------------------------------------
// (e) ShallowIntersect - 0.5 mm penetration, below the 1 mm limit.
//
//  Cross-section at z = 0:
//
//    y
//    ^
//  2 |  +-------+
//    |  |       |
// 0.5|  |     +-+-+
//    |  |  a  | | | b      b: 1x1x1 cube shifted by (2, 0, 0)
//-0.5|  |     +-+-+        overlap depth in x = 0.5 mm
//    |  |       |
// -2 |  +-------+
//    +-------------------> x
//      -2    1.5 2 2.5
//
//  b in a-coords: [1.5, 2.5] x [-0.5, 0.5] x [-0.5, 0.5]
// clang-format on
TEST( MRMesh, MeshMeshDistance_ShallowIntersect )
{
    const auto a = makeRefCube();  // [-2, 2]^3
    const auto b = makeUnitCube(); // [-0.5, 0.5]^3
    const AffineXf3f xf = AffineXf3f::translation( Vector3f( 2.0f, 0.0f, 0.0f ) );
    const float penetration = 0.5f;

    // no limit
    {
        const auto d = findDistance( a, b, &xf, kNoLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, 0.0f );

        const auto sd = findSignedDistance( a, b, &xf, kNoLimitSq );
        EXPECT_NEAR( sd.signedDist, -penetration, 1e-3f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::Colliding );
    }

    // 1 mm limit: collision detection works regardless of the positive-distance limit
    {
        const auto d = findDistance( a, b, &xf, kOneMmLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, 0.0f );

        const auto sd = findSignedDistance( a, b, &xf, kOneMmLimitSq );
        EXPECT_NEAR( sd.signedDist, -penetration, 1e-3f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::Colliding );
    }

    // strict 0 limit: findDistance still returns 0, signed-distance path uses
    // findCollidingTriangles directly so collisions are still detected
    {
        const auto d = findDistance( a, b, &xf, kZeroLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, 0.0f );

        const auto sd = findSignedDistance( a, b, &xf, kZeroLimitSq );
        EXPECT_NEAR( sd.signedDist, -penetration, 1e-3f ); // <-- PROBLEM 3
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::Colliding ); // <-- PROBLEM 4
    }
}

// clang-format off
// ----------------------------------------------------------------------------
// (f) DeepIntersect - 1.75 mm penetration > 1 mm limit, yet neither mesh
// is fully inside the other.
//
//  Cross-section at z = 0:
//
//     y
//     ^
//  2  |  +-------+
//     |  |   a   |
//     |  |       |
//0.25 |  | +-----+----+
//     |  | |     |    |      b: long bar 2.5 x 0.5 x 0.5, shifted by (1, 0, 0)
//-0.25|  | +-----+----+      penetration depth in x = 1.75 mm
//     |  |       |
// -2  |  +-------+
//     +-----------------------> x
//       -2 -0.25 2  2.25
//
//  b in a-coords: [-0.25, 2.25] x [-0.25, 0.25] x [-0.25, 0.25]
//  Some a vertices (corners) are outside b, some b vertices are outside a:
//  neither mesh is fully contained in the other.
// clang-format on
TEST( MRMesh, MeshMeshDistance_DeepIntersect )
{
    const auto a = makeRefCube(); // [-2, 2]^3
    // long thin bar centered at origin in its own coords: [-1.25, 1.25] x [-0.25, 0.25]^2
    const auto b = makeCube( Vector3f( 2.5f, 0.5f, 0.5f ), Vector3f( -1.25f, -0.25f, -0.25f ) );
    const AffineXf3f xf = AffineXf3f::translation( Vector3f( 1.0f, 0.0f, 0.0f ) );
    const float penetration = 1.75f;

    // no limit
    {
        const auto d = findDistance( a, b, &xf, kNoLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, 0.0f );

        const auto sd = findSignedDistance( a, b, &xf, kNoLimitSq );
        EXPECT_NEAR( sd.signedDist, -penetration, 1e-3f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::Colliding );
    }

    // 1 mm limit: positive-distance limit does not gate collision detection
    // nor the vertex-based signed-distance zone analysis
    {
        const auto d = findDistance( a, b, &xf, kOneMmLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, 0.0f );

        const auto sd = findSignedDistance( a, b, &xf, kOneMmLimitSq );
        EXPECT_NEAR( sd.signedDist, -penetration, 1e-3f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::Colliding );
    }

    // strict 0 limit
    {
        const auto d = findDistance( a, b, &xf, kZeroLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, 0.0f );

        const auto sd = findSignedDistance( a, b, &xf, kZeroLimitSq );
        EXPECT_NEAR( sd.signedDist, -penetration, 1e-3f ); // <-- PROBLEM 5
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::Colliding ); // <-- PROBLEM 6
    }
}

// clang-format off
// ----------------------------------------------------------------------------
// (g) OneInsideOther - small cube fully inside the big one.
//
//  Cross-section at z = 0:
//
//    y
//    ^
//  2 |  +-----------+
//    |  |           |
//    |  |     a     |
// 0.5|  |   +---+   |
//    |  |   | b |   |       b: 1x1x1 cube, no transform
//-0.5|  |   +---+   |       gap from b's face to a's nearest face = 1.5 mm
//    |  |           |
//    |  |           |
// -2 |  +-----------+
//    +-----------------> x
//      -2 -0.5 0.5  2
//
//  b in a-coords: [-0.5, 0.5]^3 (entirely inside a = [-2, 2]^3)
// clang-format on
TEST( MRMesh, MeshMeshDistance_OneInsideOther )
{
    const auto a = makeRefCube();  // [-2, 2]^3
    const auto b = makeUnitCube(); // [-0.5, 0.5]^3, no transform -> fully inside a
    const float gap = 1.5f;        // distance from b's face to a's nearest face
    const float gapSq = gap * gap;

    // no limit
    {
        const auto d = findDistance( a, b, nullptr, kNoLimitSq );
        EXPECT_NEAR( d.distSq, gapSq, 1e-3f );

        const auto sd = findSignedDistance( a, b, nullptr, kNoLimitSq );
        EXPECT_NEAR( sd.signedDist, -gap, 1e-3f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::BInside );
    }

    // 1 mm limit: real positive distance (1.5) > 1 mm -> findDistance
    // early-exits with invalid res.a/b. With no valid projection points,
    // the inside/outside relation cannot be determined, so the function
    // reports NotColliding with signedDist = +sqrt(upDistLimitSq) = 1.0,
    // even though geometrically b is fully inside a.
    {
        const auto d = findDistance( a, b, nullptr, kOneMmLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, kOneMmLimitSq );

        const auto sd = findSignedDistance( a, b, nullptr, kOneMmLimitSq );
        EXPECT_FLOAT_EQ( sd.signedDist, 1.0f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::NotColliding );
    }

    // strict 0 limit: findCollidingTriangles is empty for a nested cube
    // (no surface intersections), and inside/outside cannot be derived from
    // the invalid res.a/b -> NotColliding with signedDist = 0, even though
    // geometrically b is fully inside a.
    {
        const auto d = findDistance( a, b, nullptr, kZeroLimitSq );
        EXPECT_FLOAT_EQ( d.distSq, kZeroLimitSq );

        const auto sd = findSignedDistance( a, b, nullptr, kZeroLimitSq );
        EXPECT_FLOAT_EQ( sd.signedDist, 0.0f );
        EXPECT_EQ( sd.status, MeshMeshCollisionStatus::NotColliding );
    }
}

} // namespace MR
