#include "MRTriangleIntersection.h"
#include "MRGTest.h"

namespace
{
    constexpr MR::Triangle3f triangle1 {
        MR::Vector3f { -30.f, -52.f, 0.f },
        MR::Vector3f { +30.f, -52.f, 0.f },
        MR::Vector3f { 0.f, 0.f, 0.f },
    };
    constexpr MR::Triangle3f triangle2( float x, float y, float z )
    {
        constexpr MR::Triangle3f origin {
            MR::Vector3f { -30.f, 0.f, -52.f },
            MR::Vector3f { +30.f, 0.f, -52.f },
            MR::Vector3f { 0.f, 0.f, 0.f },
        };
        return {
            origin[0] + MR::Vector3f { x, y, z },
            origin[1] + MR::Vector3f { x, y, z },
            origin[2] + MR::Vector3f { x, y, z },
        };
    }
    constexpr MR::Triangle3f triangle3 {
        MR::Vector3f { 0.f, 0.f, -30.f },
        MR::Vector3f { 0.f, 0.f, +30.f },
        MR::Vector3f { 0.f, -52.f, 0.f },
    };
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
    // shared vertex
      TrianglesIntersectParameters { false, triangle1, triangle2( 0.f, 0.f, 0.f ) }
    // one triangle doesn't intersect the other triangle's plane
    , TrianglesIntersectParameters { false, triangle1, triangle2( 0.f, -26.f, -13.f ) }
    // one triangle doesn't intersect the other triangle
    , TrianglesIntersectParameters { false, triangle1, triangle2( +26.f, -26.f, +13.f ) }
    // edge-tri intersections
    , TrianglesIntersectParameters { true, triangle1, triangle2( 0.f, -26.f, +13.f ) }
    // mutual edge-tri intersections
    , TrianglesIntersectParameters { true, triangle1, triangle2( +13.f, -26.f, +13.f ) }
    // two edge-edge intersections
    , TrianglesIntersectParameters { true, triangle1, triangle2( 0.f, -26.f, +26.f ) }
    // two edge-vertex intersections
    , TrianglesIntersectParameters { true, triangle1, triangle3 }
) );
}
