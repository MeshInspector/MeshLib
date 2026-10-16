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

/// the line that an edge goes along inside a cut triangle, given for one direction of the edge
struct PreciseCutLine
{
    /// for a cut: the vertices of the other triangle, whose plane crosses the cut triangle along the line
    std::array<PreciseVertCoords, 3> plane;

    /// for a cut: whether the edge goes along cross( normal of the cut triangle, normal of the plane )
    bool alongCross = false;

    /// for a side of the cut triangle: the index i of the side, which goes from vertex i of the triangle to vertex i+1,
    /// counterclockwise around the triangle's normal; so does the edge, which has the triangle on its left;
    /// -1 for a cut
    int side = -1;

    [[nodiscard]] bool isSide() const { return side >= 0; }
};

/// exact predicates on the points inside one triangle cut along its intersections with other triangles, for a sweep
/// line over the triangle; they never contradict the boolean's exact predicates, because every point gets the same
/// simulation-of-simplicity perturbation by vertex ids as in \ref orient3d, and a point is on the side of a cut where
/// the boolean's predicates put it;
/// the sweep goes over the projection of the perturbed triangle on the plane of the two axes least aligned with its normal,
/// meeting the points in the order of one linear functional, which is nearly the first projected coordinate
class MR_BIND_IGNORE CutTrianglePredicates
{
public:
    /// starts a new triangle, given counterclockwise around its normal, forgetting all points but keeping the memory;
    /// returns false if the exact normal of the triangle is zero
    MRMESH_API bool reset( const std::array<PreciseVertCoords, 3> & tri );

    /// sets point v: a vertex of the triangle, or a crossing of one of its sides with another triangle,
    /// or a crossing of an edge of another triangle with this triangle;
    /// returns false if the point is not defined, because its segment is parallel to the plane of its triangle
    MRMESH_API bool addPoint( VertId v, const PreciseCutPoint & p );

    /// sets point v at the crossing of two cuts;
    /// returns false if their planes and the plane of the triangle have no single common point
    MRMESH_API bool addCrossing( VertId v, const PreciseCutLine & a, const PreciseCutLine & b, const ConvertToFloatVector & toFloat );

    /// whether point p comes before point q along the sweep
    [[nodiscard]] MRMESH_API bool less( VertId p, VertId q ) const;

    /// the same as less for two points that coincide before the perturbation, decided by expanding the perturbation alone,
    /// while less decides some of them by faster rules; nullopt if the points coincide after the perturbation too
    [[nodiscard]] MRMESH_API std::optional<bool> lessByPerturbation( VertId p, VertId q ) const;

    /// whether point p is to the left of line l, taken in its direction, or in the opposite one if reversed;
    /// p must not be a crossing nor lie on the line
    [[nodiscard]] MRMESH_API bool leftOf( const PreciseCutLine & l, bool reversed, VertId p ) const;

    /// for an edge going along line l (against it if reversed) from point o to point x: whether the triangle of x, o and a point
    /// far behind o against the sweep is counterclockwise
    [[nodiscard]] MRMESH_API bool ccwFromBehind( const PreciseCutLine & l, bool reversed ) const;

    /// whether point v was set by addCrossing
    [[nodiscard]] bool isCrossing( VertId v ) const { return points_[v].crossing >= 0; }

    /// the position of crossing v, rounded in the integer coordinates and converted by toFloat;
    /// it is the same in each of the three triangles whose planes meet there
    [[nodiscard]] MRMESH_API Vector3f crossingPos( VertId v ) const;

    /// whether a predicate met points it cannot tell apart (e.g. two points with the same definition), or a point
    /// was not defined; the answers of the predicates are arbitrary then, and the caller must discard them
    [[nodiscard]] bool failed() const { return failed_.load( std::memory_order_relaxed ); }

private:
    struct Crossing
    {
        std::array<PreciseVertCoords, 3> a, b; // the planes of the two cuts
        FastInt256 nx, ny, d; // the projected coordinates are nx / d and ny / d, with d > 0
        Vector3f pos;
    };

    struct Point
    {
        PreciseCutPoint def; // not used for a crossing
        FastInt128 oa, ob; // for an edge-triangle point: the volumes of its triangle with the ends of its segment
        int crossing = -1; // for a crossing: its index in crossings_
    };

    /// a projected coordinate as a fraction with a positive denominator
    struct Fraction
    {
        FastInt256 n, d;
    };

    Fraction exactCoord_( VertId v, int axis ) const;
    int compareExact_( VertId p, VertId q, int axis ) const;
    bool lessCoincident_( VertId p, VertId q ) const;
    int sweepSign_( const Vector3i64 & u ) const;
    bool sideOfPlane_( const std::array<PreciseVertCoords, 3> & plane, const PreciseCutPoint & p ) const;
    void fail_() const { failed_.store( true, std::memory_order_relaxed ); }

    std::array<PreciseVertCoords, 3> tri_;
    Vector3i64 normal_;
    int kx_ = 0, ky_ = 1; // the projection axes, in this order ccw around the normal
    Vector<Vector2d, VertId> approx_; // the projected coordinates of every point, rounded, to decide most comparisons
    Vector<Point, VertId> points_;
    std::vector<Crossing> crossings_;
    mutable std::atomic<bool> failed_{ false };
    // the orders of the pairs of points found by expanding the perturbation, which is slow, while a sweep compares the same points often;
    // the key is ( p << 32 ) | q for p < q; a sweep sorts the points in parallel, hence the mutex
    mutable HashMap<std::uint64_t, bool> perturbedLess_;
    mutable std::mutex perturbedLessMutex_;
};

/// \}

} // namespace MR
