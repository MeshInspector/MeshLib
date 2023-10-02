#include <MRMesh/MRMeshFwd.h>

#include <MRMesh/MRTriangleIntersection.h>
#include <MRMesh/MRGTest.h>

namespace
{
    // triangle lying on XY plane
    constexpr MR::Triangle3f T1 {
        MR::Vector3f { -30.f, -52.f, 0.f },
        MR::Vector3f { +30.f, -52.f, 0.f },
        MR::Vector3f { 0.f, 0.f, 0.f },
    };
    // triangle lying on XZ plane
    constexpr MR::Triangle3f T2 {
        MR::Vector3f { -30.f, 0.f, -52.f },
        MR::Vector3f { +30.f, 0.f, -52.f },
        MR::Vector3f { 0.f, 0.f, 0.f },
    };
    // triangle lying on YZ plane
    constexpr MR::Triangle3f T3 {
        MR::Vector3f { 0.f, 0.f, -30.f },
        MR::Vector3f { 0.f, 0.f, +30.f },
        MR::Vector3f { 0.f, -52.f, 0.f },
    };
    constexpr MR::Triangle3f shift( const MR::Triangle3f& origin, float x, float y, float z )
    {
        return {
            origin[0] + MR::Vector3f { x, y, z },
            origin[1] + MR::Vector3f { x, y, z },
            origin[2] + MR::Vector3f { x, y, z },
        };
    }
}

namespace MR
{
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

using TrianglesIntersectParameters = std::tuple<bool, Triangle3f, Triangle3f>;
class TrianglesIntersectTestFixture : public testing::TestWithParam<TrianglesIntersectParameters> { };

/// check all vertex triplet configurations
TEST_P( TrianglesIntersectTestFixture, TrianglesIntersect )
{
    using VertexOrder = std::array<size_t, 3>;
    constexpr std::array<VertexOrder, 6> permutations {
        VertexOrder { 0, 1, 2 },
        VertexOrder { 0, 2, 1 },
        VertexOrder { 1, 0, 2 },
        VertexOrder { 1, 2, 0 },
        VertexOrder { 2, 0, 1 },
        VertexOrder { 2, 1, 0 },
    };

    const auto& [result, triA, triB] = GetParam();
    for ( const auto& orderA : permutations )
    {
        for ( const auto& orderB : permutations )
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

} // namespace MR
