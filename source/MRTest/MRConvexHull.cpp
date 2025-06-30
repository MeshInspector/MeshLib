#include "MRMesh/MRConvexHull.h"

#include "MRMesh/MRGTest.h"

namespace MR
{

// Test case for an empty set of points
TEST(MakeConvexHullTest, EmptyInput) {
    std::vector<Vector2f> points;
    std::vector<Vector2f> hull = makeConvexHull(points);
    ASSERT_TRUE(hull.empty());
}

// Test case for a single point
TEST(MakeConvexHullTest, SinglePoint) {
    std::vector<Vector2f> points = {{1.0f, 1.0f}};
    std::vector<Vector2f> hull = makeConvexHull(points);
    EXPECT_EQ(hull, points);
}

// Test case for two points
TEST(MakeConvexHullTest, TwoPoints) {
    std::vector<Vector2f> points = {{2.0f, 2.0f}, {1.0f, 1.0f} };
    std::vector<Vector2f> hull = makeConvexHull(points);
    ASSERT_EQ(hull.size(), 2);
    // The hull should start from a minimal coords point.
    EXPECT_EQ(hull[0], points[1]);
    EXPECT_EQ(hull[1], points[0]);
}

// Test case for three collinear points
TEST(MakeConvexHullTest, ThreeCollinearPoints) {
    std::vector<Vector2f> points = {{1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}};
    std::vector<Vector2f> hull = makeConvexHull(points);
    ASSERT_EQ(hull.size(), 2);
    // The hull should only contain the two extreme points
    EXPECT_EQ(hull[0], points[0]);
    EXPECT_EQ(hull[1], points[2]);
}

// Test case for a simple square
TEST(MakeConvexHullTest, SimpleSquare) {
    std::vector<Vector2f> points = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    std::vector<Vector2f> hull = makeConvexHull(points);
    ASSERT_EQ(hull.size(), 4);

    // The expected hull in counter-clockwise order
    std::vector<Vector2f> expected_hull = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
    EXPECT_EQ(hull, expected_hull);
}

// Test case with points inside the hull
TEST(MakeConvexHullTest, PointsInside) {
    std::vector<Vector2f> points = {
        {0.0f, 0.0f}, {5.0f, 0.0f}, {0.0f, 5.0f}, {5.0f, 5.0f}, // The corners of the hull
        {1.0f, 1.0f}, {2.0f, 3.0f}, {4.0f, 2.0f}               // Points inside
    };
    std::vector<Vector2f> hull = makeConvexHull(points);
    ASSERT_EQ(hull.size(), 4);

    std::vector<Vector2f> expected_hull = {{0.0f, 0.0f}, {5.0f, 0.0f}, {5.0f, 5.0f}, {0.0f, 5.0f}};
    EXPECT_EQ(hull, expected_hull);
}

// Test case with duplicate points
TEST(MakeConvexHullTest, DuplicatePoints) {
    std::vector<Vector2f> points = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}};
    std::vector<Vector2f> hull = makeConvexHull(points);
    ASSERT_EQ(hull.size(), 3);

    std::vector<Vector2f> expected_hull = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
    EXPECT_EQ(hull, expected_hull);
}

// A more complex test case with various points
TEST(MakeConvexHullTest, ComplexShape) {
    std::vector<Vector2f> points = {
        {0, 3}, {1, 1}, {2, 2}, {4, 4},
        {0, 0}, {1, 2}, {3, 1}, {3, 3}
    };
    std::vector<Vector2f> hull = makeConvexHull(points);
    ASSERT_EQ(hull.size(), 4);

    std::vector<Vector2f> expected_hull = {{0, 0}, {3, 1}, {4, 4}, {0, 3}};
    EXPECT_EQ(hull, expected_hull);
}

} // namespace MR
