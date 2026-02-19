#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRTriangleIntersection.h>
#include <MRMesh/MRGTest.h>


namespace MR
{

namespace
{

// triangle lying on XY plane
constexpr Triangle3f T1 {
    Vector3f { -30.f, -52.f, 0.f },
    Vector3f { +30.f, -52.f, 0.f },
    Vector3f { 0.f, 0.f, 0.f },
};

// triangle lying on XZ plane
constexpr Triangle3f T2 {
    Vector3f { -30.f, 0.f, -52.f },
    Vector3f { +30.f, 0.f, -52.f },
    Vector3f { 0.f, 0.f, 0.f },
};

// triangle lying on YZ plane
constexpr Triangle3f T3 {
    Vector3f { 0.f, 0.f, -30.f },
    Vector3f { 0.f, 0.f, +30.f },
    Vector3f { 0.f, -52.f, 0.f },
};
constexpr Triangle3f shift( const Triangle3f& origin, float x, float y, float z )
{
    return {
        origin[0] + Vector3f { x, y, z },
        origin[1] + Vector3f { x, y, z },
        origin[2] + Vector3f { x, y, z },
    };
}

using VertexOrder = std::array<size_t, 3>;
constexpr std::array<VertexOrder, 6> triPermutations
{
    VertexOrder { 0, 1, 2 },
    VertexOrder { 0, 2, 1 },
    VertexOrder { 1, 0, 2 },
    VertexOrder { 1, 2, 0 },
    VertexOrder { 2, 0, 1 },
    VertexOrder { 2, 1, 0 },
};

} //anonymous namespace

TEST( MRMesh, TriangleSegmentIntersectFloat )
{
    Vector3f a{2,  1, 0};
    Vector3f b{-2,  1, 0};
    Vector3f c{0, -2, 0};

    Vector3f d{0, 0, -1};
    Vector3f e{0, 0,  1};

    bool intersection = doTriangleSegmentIntersect( a, b, c, d, e );

    EXPECT_TRUE( intersection );
}

TEST( MRMesh, PointTriangleIntersectFloat )
{
    EXPECT_TRUE( isPointInPlane( T1[0], T1[0], T1[1], T1[2] ) );
    EXPECT_TRUE( isPointInPlane( ( T1[0] + T1[1] + T1[2] ) / 3.0f, T1[0], T1[1], T1[2] ) );
    EXPECT_TRUE( isPointInPlane( ( T1[0] + T1[1] ) / 2.0f, T1[0], T1[1], T1[2] ) );
    EXPECT_TRUE( isPointInPlane( Vector3f( 100, 100, 0 ), T1[0], T1[1], T1[2] ) );
    EXPECT_FALSE( isPointInPlane( Vector3f( 0, 0, 1 ), T1[0], T1[1], T1[2] ) );

    EXPECT_TRUE( isPointInTriangle( T1[0], T1[0], T1[1], T1[2] ) );
    EXPECT_TRUE( isPointInTriangle( ( T1[0] + T1[1] + T1[2] ) / 3.0f, T1[0], T1[1], T1[2] ) );
    EXPECT_TRUE( isPointInTriangle( ( T1[0] + T1[1] ) / 2.0f, T1[0], T1[1], T1[2] ) );
    EXPECT_FALSE( isPointInTriangle( Vector3f( 100, 100, 0 ), T1[0], T1[1], T1[2] ) );
    EXPECT_FALSE( isPointInTriangle( Vector3f( 0, 0, 1 ), T1[0], T1[1], T1[2] ) );
}

using TrianglesIntersectParameters = std::tuple<bool, Triangle3f, Triangle3f>;
class TrianglesIntersectTestFixture : public testing::TestWithParam<TrianglesIntersectParameters> { };

/// check all vertex triplet configurations
TEST_P( TrianglesIntersectTestFixture, TrianglesIntersect )
{
    const auto& [result, triA, triB] = GetParam();
    for ( const auto& orderA : triPermutations )
    {
        for ( const auto& orderB : triPermutations )
        {
            const auto& a = triA[orderA[0]];
            const auto& b = triA[orderA[1]];
            const auto& c = triA[orderA[2]];
            const auto& d = triB[orderB[0]];
            const auto& e = triB[orderB[1]];
            const auto& f = triB[orderB[2]];
            EXPECT_EQ( result, doTrianglesIntersect( a, b, c, d, e, f ) );
        }
    }
}

INSTANTIATE_TEST_SUITE_P( MRMesh, TrianglesIntersectTestFixture, testing::Values(
    /*
     * no intersection
     */
    // T1 and T2 share a vertex
      TrianglesIntersectParameters { false, T1, T2 }
    // T1 doesn't intersect T2 plane
    , TrianglesIntersectParameters { false, T1, shift( T2, 0.f, -26.f, -13.f ) }
    // T1 vertex lies on T2 side
    , TrianglesIntersectParameters { false, T1, shift( T2, 0.f, -52.f, 0.f ) }
    // T1 and T2 share a side
    , TrianglesIntersectParameters { false, T1, shift( T2, 0.f, -52.f, +52.f ) }
    // T1 intersects T2 plane but doesn't intersect T2
    , TrianglesIntersectParameters { false, T1, shift( T2, +26.f, -26.f, +13.f ) }
    // T1 side intersect T3 side
    , TrianglesIntersectParameters { false, T1, shift( T3, 0.f, -52.f, 0.f ) }

    /*
     * intersection
     */
    // two T1 sides pass through T2
    , TrianglesIntersectParameters { true, T1, shift( T2, 0.f, -26.f, +13.f ) }
    // T1 side passes through T2, T2 side passes through T1
    , TrianglesIntersectParameters { true, T1, shift( T2, +13.f, -26.f, +13.f ) }
    // two T1 sides intersect corresponding T2 sides
    , TrianglesIntersectParameters { true, T1, shift( T2, 0.f, -26.f, +26.f ) }
    // T1 vertex lies on T3 side, T3 vertex lies on T1 side
    , TrianglesIntersectParameters { true, T1, T3 }
) );

using TrianglesOverlapParameters = std::tuple<bool, Vector2f, Vector2f, Vector2f, Vector2f, Vector2f, Vector2f>;
class TrianglesOverlapTestFixture : public testing::TestWithParam<TrianglesOverlapParameters> { };

/// check all vertex triplet configurations
TEST_P( TrianglesOverlapTestFixture, TrianglesOverlap )
{
    Vector2f triA[3];
    Vector2f triB[3];
    const auto& [result,_a,_b,_c,_d,_e,_f] = GetParam();
    triA[0] = _a; triA[1] = _b; triA[2] = _c;
    triB[0] = _d; triB[1] = _e; triB[2] = _f;
    for ( const auto& orderA : triPermutations )
    {
        for ( const auto& orderB : triPermutations )
        {
            const auto& a = triA[orderA[0]];
            const auto& b = triA[orderA[1]];
            const auto& c = triA[orderA[2]];
            const auto& d = triB[orderB[0]];
            const auto& e = triB[orderB[1]];
            const auto& f = triB[orderB[2]];
            EXPECT_EQ( result, doTrianglesOverlap( a, b, c, d, e, f ) );
            EXPECT_EQ( result, doTrianglesOverlap( d, e, f, a, b, c ) ); // test symmetry
        }
    }
}

INSTANTIATE_TEST_SUITE_P( MRMesh, TrianglesOverlapTestFixture, testing::Values(
    /*
     * no overlap
     */
    // both degenerated, far away
      TrianglesOverlapParameters{ false, Vector2f(), Vector2f(), Vector2f(), Vector2f(5,5), Vector2f( 5,5 ), Vector2f( 5,5 ) }
    // general case, both valid, far away
    , TrianglesOverlapParameters{ false, Vector2f(), Vector2f( 0,1 ), Vector2f( 1,1 ), Vector2f( 5,5 ), Vector2f( 5,6 ), Vector2f( 6,6 ) }
    // one degenerated one valid far away
    , TrianglesOverlapParameters{ false, Vector2f(), Vector2f(), Vector2f(), Vector2f( 5,5 ), Vector2f( 5,6 ), Vector2f( 6,6 ) }

    /*
     * overlap
     */
    // both degenerated, same coord
    , TrianglesOverlapParameters{ true, Vector2f(), Vector2f(), Vector2f(), Vector2f(), Vector2f(), Vector2f() }
    // one degenerated toching
    , TrianglesOverlapParameters{ true, Vector2f(), Vector2f(), Vector2f(), Vector2f(), Vector2f( 0,1 ), Vector2f( 1,1 ) }
    // both valid touching
    , TrianglesOverlapParameters{ true, Vector2f(), Vector2f( 0,1 ), Vector2f( 1,1 ), Vector2f( -0.5f,0 ), Vector2f( 0.5f,0 ), Vector2f( 0,-1 ) }
    // both valid one point inside
    , TrianglesOverlapParameters{ true, Vector2f(), Vector2f( 0,1 ), Vector2f( 1,1 ), Vector2f( -0.5f,0.5f ), Vector2f( 0.5f,0 ), Vector2f( 0,-1 ) }
    // both valid two points inside
    , TrianglesOverlapParameters{ true, Vector2f(), Vector2f( 0,1 ), Vector2f( 1,1 ), Vector2f( -0.5f,0.5f ), Vector2f( 1,1.5f ), Vector2f( 0,-1 ) }
    // both valid full inside
    , TrianglesOverlapParameters{ true, Vector2f(), Vector2f( 0,1 ), Vector2f( 1,1 ), Vector2f( -0.5f,1 ), Vector2f( 2,2 ), Vector2f( 0,-1 ) }
    // both valid none inside (3 inters)
    , TrianglesOverlapParameters{ true, Vector2f(), Vector2f( 0,1 ), Vector2f( 1,1 ), Vector2f( -0.25f,0.5f ), Vector2f( 0.5,1.25f ), Vector2f( 1,0 ) }
    // both valid none inside (2 inters)
    , TrianglesOverlapParameters{ true, Vector2f(), Vector2f( 0,1 ), Vector2f( 1,1 ), Vector2f( -0.25f,0.5f ), Vector2f( 0.5,1.25f ), Vector2f( 1,1.25f ) }
) );

TEST( MRMesh, DegenerateTrianglesIntersect )
{
    Vector3f triA[3] =
    {
        { -24.5683002f, -17.7052994f, -21.3701000f },
        { -24.6611996f, -17.7504997f, -21.3423004f },
        { -24.6392994f, -17.7071991f, -21.3542995f }
    };

    Vector3f triB[3] =
    {
        { -24.5401993f, -17.7504997f, -21.3390007f },
        { -24.5401993f, -17.7504997f, -21.3390007f },
        { -24.5862007f, -17.7504997f, -21.3586998f }
    };

    for ( const auto& orderA : triPermutations )
    {
        for ( const auto& orderB : triPermutations )
        {
            const auto& a = triA[orderA[0]];
            const auto& b = triA[orderA[1]];
            const auto& c = triA[orderA[2]];
            const auto& d = triB[orderB[0]];
            const auto& e = triB[orderB[1]];
            const auto& f = triB[orderB[2]];

            // in float arithmetic this test fails unfortunately
            EXPECT_FALSE( doTrianglesIntersect( Vector3d{a}, Vector3d{b}, Vector3d{c}, Vector3d{d}, Vector3d{e}, Vector3d{f} ) );
            EXPECT_FALSE( doTrianglesIntersect( Vector3d{d}, Vector3d{e}, Vector3d{f}, Vector3d{a}, Vector3d{b}, Vector3d{c} ) );

            EXPECT_FALSE( doTrianglesIntersectExt( a, b, c, d, e, f ) );
            EXPECT_FALSE( doTrianglesIntersectExt( d, e, f, a, b, c ) );
        }
    }
}

} // namespace MR
