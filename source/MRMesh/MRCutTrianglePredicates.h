#pragma once

#include "MRPrecisePredicates3.h"
#include "MRVector.h"
#include "MRVector2.h"
#include <memory>
#include <optional>

namespace MR
{

/// \addtogroup MathGroup
/// \{

/// a point inside a triangle of a mesh cut along its intersections with other triangles, defined by mesh vertices
/// in the exact coordinates that the boolean's predicates take, with the same ids: either a vertex, or the point
/// where a segment crosses the plane of a triangle (an edge-triangle intersection found by \ref doTriangleSegmentIntersect)
struct MR_BIND_IGNORE PreciseCutPoint
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
/// (dropping the axis of the largest component of the normal, the first one of equal components, or the axis of the leading
/// term of the perturbed normal if the exact one is zero), meeting the points in the order of one linear functional,
/// which is nearly the first projected coordinate
class MR_BIND_IGNORE CutTrianglePredicates
{
public:
    MRMESH_API CutTrianglePredicates();
    MRMESH_API CutTrianglePredicates( CutTrianglePredicates && ) noexcept;
    MRMESH_API CutTrianglePredicates & operator =( CutTrianglePredicates && ) noexcept;
    MRMESH_API ~CutTrianglePredicates();

    /// starts a new triangle with three distinct vertex ids: forgets all points, the orders and orientations found by
    /// the slower rules, and the failure, keeping the memory of the points for the next triangle; the vertices may come
    /// in any order, since the projection shows them counterclockwise around their right-hand normal, which puts
    /// the triangle on the left of every side i, from tri[i] to tri[i+1]
    MRMESH_API void reset( const std::array<PreciseVertCoords, 3> & tri );

    /// sets point v: a vertex of the triangle, or a crossing of one of its sides with another triangle,
    /// or a crossing of an edge of another triangle with this triangle, as \ref doTriangleSegmentIntersect finds them;
    /// a segment in the plane of its triangle crosses it after the perturbation, at a point that tends to an end of the segment
    /// or to a point inside it; setting a point again forgets the orders found for its old definition;
    /// a vertex point that is not a corner of the triangle, repeated ids, a segment with an end on a vertex of its own triangle,
    /// or a segment that does not cross the plane of its triangle make the predicates fail
    MRMESH_API void addPoint( VertId v, const PreciseCutPoint & p );

    /// whether point p comes before point q along the sweep
    [[nodiscard]] MRMESH_API bool less( VertId p, VertId q ) const;

    /// the signs of the projected coordinates of q - p found by expanding the perturbation alone, as the sweep sees them:
    /// a coordinate whose leading term has a larger degree than that of the other one is zero; this is slow,
    /// while less and ccwFromBehind decide most pairs by faster rules; nullopt if the points coincide after the perturbation too
    [[nodiscard]] MRMESH_API std::optional<Vector2i> directionByPerturbation( VertId p, VertId q ) const;

    /// whether triangle abc is counterclockwise in the projection: decided by the exact positions of the points,
    /// or by ccwByLines if they are on one line before the perturbation, or else by expanding the perturbation,
    /// so any three points are decided; as the sweep asks about the same triangles often, the answers of the slower rules
    /// are kept till reset
    [[nodiscard]] MRMESH_API bool ccw( VertId a, VertId b, VertId c ) const;

    /// the same as ccw, decided by a line through two of the points known from their definitions, as the ends of every edge
    /// of the cut triangle are: one side of the triangle, or the plane of another triangle (by which the cut along the line goes),
    /// or the plane through a vertex and a crossing (at selfBoolean's folds), which ccw needs only for points on one line
    /// before the perturbation; the points on one side stay on one line after the perturbation, and they are taken
    /// as bulging inward a little, so three of them make a counterclockwise triangle if they come in the order
    /// from vertex i+1 to vertex i of side i, up to a cyclic shift;
    /// nullopt if no two of the points are on a known line
    [[nodiscard]] MRMESH_API std::optional<bool> ccwByLines( VertId a, VertId b, VertId c ) const;

    /// the same as ccw found by expanding the perturbation alone, which is slow, while ccw decides most triangles by faster rules;
    /// nullopt if the points are on one line after the perturbation too, as the points of one side are
    [[nodiscard]] MRMESH_API std::optional<bool> ccwByPerturbation( VertId a, VertId b, VertId c ) const;

    /// whether the triangle of a point far behind pivot against the sweep, b and pivot is counterclockwise
    [[nodiscard]] MRMESH_API bool ccwFromBehind( VertId b, VertId pivot ) const;

    /// whether a predicate met points it cannot tell apart (two points with the same definition, or three points off the sides
    /// on one line also after the perturbation, which needs such points), or the input broke a precondition (see addPoint);
    /// the answers of the predicates are arbitrary then, and the caller must discard them
    [[nodiscard]] MRMESH_API bool failed() const;

private:
    /// the line through two points: a side of the triangle, or its crossing with the plane through three vertices
    struct Line
    {
        int side = -1; // -1 for a plane
        std::array<PreciseVertCoords, 3> plane;
    };

    /// the points with their exact data, the orders and orientations found by the slower rules, and the failure flag;
    /// defined in the .cpp file
    struct Impl;

    int compareExact_( VertId p, VertId q, bool y ) const;
    int exactOrient_( VertId a, VertId b, VertId c ) const;
    bool ccwExact_( VertId a, VertId b, VertId c ) const;
    int orientSlow_( VertId a, VertId b, VertId c ) const;
    std::optional<bool> ccwBySides_( VertId a, VertId b, VertId c ) const;
    std::optional<bool> ccwByPlanes_( VertId a, VertId b, VertId c ) const;
    bool lessCoincident_( VertId p, VertId q ) const;
    std::optional<Vector2i> directionFromSide_( VertId p, VertId q ) const;
    std::optional<Vector2i> perturbedDirection_( VertId p, VertId q ) const;
    Vector2i segmentDirection_( const PreciseVertCoords & a, const PreciseVertCoords & b ) const;
    std::optional<Line> commonLine_( VertId p, VertId q, bool withVertex ) const;
    Vector2i lineDirection_( const Line & l ) const;
    bool along_( const Vector2i & dir, VertId p, VertId q ) const;
    bool sideOfPlane_( const std::array<PreciseVertCoords, 3> & plane, VertId x ) const;
    void fail_() const;

    std::array<PreciseVertCoords, 3> tri_;
    Vector3i64 normal_;
    int kx_ = 0, ky_ = 1; // the projection axes, in this order ccw around the normal
    Vector<Vector2d, VertId> approx_; // the projected coordinates of every point, rounded, to decide most comparisons
    std::unique_ptr<Impl> impl_;
};

/// \}

} // namespace MR
