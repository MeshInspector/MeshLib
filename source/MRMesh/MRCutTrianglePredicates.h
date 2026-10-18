#pragma once

#include "MRPrecisePredicates3.h"
#include "MRFastInt.h"
#include "MRVector.h"
#include "MRVector2.h"
#include "MRphmap.h"
#include <atomic>
#include <mutex>
#include <optional>

namespace MR
{

/// \addtogroup MathGroup
/// \{

/// a point inside a triangle of a mesh cut along its intersections with other triangles, defined by mesh vertices
/// in the exact coordinates that the boolean's predicates take, with the same ids: either a vertex, or the point
/// where a segment crosses the plane of a triangle (an edge-triangle intersection found by \ref doTriangleSegmentIntersect)
struct PreciseCutPoint
{
    /// the vertex in seg[0] if seg[1].id is invalid; otherwise the ends of the segment
    std::array<PreciseVertCoords, 2> seg;

    /// the triangle whose plane the segment crosses; not used for a vertex
    std::array<PreciseVertCoords, 3> tri;

    [[nodiscard]] bool isVertex() const { return !seg[1].id.valid(); }
};

/// exact predicates on the points inside one triangle cut along its intersections with other triangles, for a sweep
/// line over the triangle; they never contradict the boolean's exact predicates, because every point gets the same
/// simulation-of-simplicity perturbation by vertex ids as in \ref orient3d, and a point is on the side of a cut where
/// the boolean's predicates put it;
/// the sweep goes over the projection of the perturbed triangle on the plane of the two axes least aligned with its normal
/// (with its perturbed normal, if the exact one is zero), meeting the points in the order of one linear functional,
/// which is nearly the first projected coordinate
class MR_BIND_IGNORE CutTrianglePredicates
{
public:
    /// starts a new triangle, given counterclockwise around its normal, forgetting all points but keeping the memory
    MRMESH_API void reset( const std::array<PreciseVertCoords, 3> & tri );

    /// sets point v: a vertex of the triangle, or a crossing of one of its sides with another triangle,
    /// or a crossing of an edge of another triangle with this triangle, as \ref doTriangleSegmentIntersect finds them;
    /// a segment in the plane of its triangle crosses it after the perturbation, at a point that tends to an end of the segment
    /// or to a point inside it
    MRMESH_API void addPoint( VertId v, const PreciseCutPoint & p );

    /// whether point p comes before point q along the sweep
    [[nodiscard]] MRMESH_API bool less( VertId p, VertId q ) const;

    /// the same as less, decided by expanding the perturbation alone, which is slow, while less decides most pairs by faster rules;
    /// nullopt if the points coincide after the perturbation too
    [[nodiscard]] MRMESH_API std::optional<bool> lessByPerturbation( VertId p, VertId q ) const;

    /// whether triangle abc is counterclockwise in the projection;
    /// two of the points must be on one line, as the ends of every edge of a sweep over the cut triangle are:
    /// on one side of the triangle, or on the plane of one other triangle (by which the cut along the line goes);
    /// the points on one side are taken as bulging outward a little, so three of them make a counterclockwise triangle
    /// if they come in this order along the side, from its vertex i to vertex i+1, up to a cyclic shift
    [[nodiscard]] MRMESH_API bool ccw( VertId a, VertId b, VertId c ) const;

    /// whether the triangle of a point far behind pivot against the sweep, b and pivot is counterclockwise;
    /// if b and pivot coincide before the perturbation, they must be on one line as in ccw
    [[nodiscard]] MRMESH_API bool ccwFromBehind( VertId b, VertId pivot ) const;

    /// whether a predicate met points it cannot tell apart (two points with the same definition), or no two points of a ccw
    /// were on one line, or the input broke a precondition (e.g. repeated ids, or a segment that does not cross the plane
    /// of its triangle); the answers of the predicates are arbitrary then, and the caller must discard them
    [[nodiscard]] bool failed() const { return failed_.load( std::memory_order_relaxed ); }

private:
    struct Point
    {
        PreciseCutPoint def;
        // for an edge-triangle point: the volumes of its triangle with the ends of its segment; if both are zero,
        // the leading coefficients of the perturbed volumes, with zero for the one of the larger degree
        FastInt128 oa, ob;
        int sides; // bit i is set if the point is on side i of the triangle, from vertex i to vertex i+1
        // no default member initializers: with them, clang takes Point as not default-constructible in Vector::resize
    };

    /// a projected coordinate as a fraction with a positive denominator
    struct Fraction
    {
        FastInt256 n, d;
    };

    /// the line through two points: a side of the triangle, or its crossing with the plane through three vertices
    struct Line
    {
        int side = -1; // -1 for a plane
        std::array<PreciseVertCoords, 3> plane;
    };

    Fraction exactCoord_( VertId v, int axis ) const;
    int compareExact_( VertId p, VertId q, int axis ) const;
    bool lessCoincident_( VertId p, VertId q ) const;
    Vector2i segmentDirection_( const PreciseVertCoords & a, const PreciseVertCoords & b ) const;
    std::optional<Line> commonLine_( VertId p, VertId q, bool withVertex ) const;
    Vector2i lineDirection_( const Line & l ) const;
    bool along_( const Vector2i & dir, VertId p, VertId q ) const;
    bool sideOfPlane_( const std::array<PreciseVertCoords, 3> & plane, const PreciseCutPoint & p ) const;
    void fail_() const { failed_.store( true, std::memory_order_relaxed ); }

    std::array<PreciseVertCoords, 3> tri_;
    Vector3i64 normal_;
    int kx_ = 0, ky_ = 1; // the projection axes, in this order ccw around the normal
    Vector<Vector2d, VertId> approx_; // the projected coordinates of every point, rounded, to decide most comparisons
    Vector<Point, VertId> points_;
    mutable std::atomic<bool> failed_{ false };
    // the orders of the pairs of points found by expanding the perturbation, which is slow, while a sweep compares the same points often;
    // the key is ( p << 32 ) | q for p < q; a sweep sorts the points in parallel, hence the mutex
    mutable HashMap<std::uint64_t, bool> perturbedLess_;
    mutable std::mutex perturbedLessMutex_;
};

/// \}

} // namespace MR
