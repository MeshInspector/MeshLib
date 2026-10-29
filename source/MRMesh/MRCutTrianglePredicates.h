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
/// where a segment crosses the plane of a triangle (an edge-triangle intersection found by \ref doTriangleSegmentIntersect),
/// or the crossing of two cuts of the triangle, where the planes of two other triangles and its own one meet;
/// the coordinates are below 2^30 by absolute value, as \ref getToIntConverter makes them, which the widths of the exact integers
/// and the error bounds of the predicates rely on, and an id stands for one vertex with the same coordinates in every point
struct MR_BIND_IGNORE PreciseCutPoint
{
    /// the vertex in seg[0] if seg[1].id is invalid; otherwise the ends of the segment; ignored for a crossing of two cuts (see tri2)
    std::array<PreciseVertCoords, 2> seg;

    /// the triangle whose plane the segment crosses, or the plane of the first cut of a crossing of two cuts; not used for a vertex
    std::array<PreciseVertCoords, 3> tri;

    /// optional for a segment that is an edge of another mesh crossing the triangle: the third vertices of the two triangles of that edge,
    /// in any order, with invalid ids if not given (as for a boundary edge); they tell apart the crossings of the edges of a multiple edge,
    /// which are defined alike otherwise (see \ref CutTrianglePredicates); ignored for a crossing of two cuts (see tri2)
    std::array<PreciseVertCoords, 2> wedge;

    /// the plane of the second cut of a crossing of two cuts, with invalid ids for the other points: a valid tri2[0].id makes the point
    /// a crossing of two cuts, whatever seg and wedge hold
    std::array<PreciseVertCoords, 3> tri2;

    [[nodiscard]] bool isCrossing() const { return tri2[0].id.valid(); }
    [[nodiscard]] bool isVertex() const { return !seg[1].id.valid() && !isCrossing(); }
};

/// exact predicates on the points inside one triangle cut along its intersections with other triangles, for a sweep
/// line over the triangle; they never contradict the boolean's exact predicates, because every point gets the same
/// simulation-of-simplicity perturbation by vertex ids as in \ref orient3d, and a point is on the side of a cut where
/// the boolean's predicates put it;
/// the crossings of a segment with the triangle defined alike (the edges of a multiple edge of another mesh), which that perturbation
/// cannot tell apart and the boolean's predicates never compare, are told apart by the triangles of their edges: each edge is shifted by
/// an amount infinitely smaller than every term of the perturbation, toward the third vertices of its triangles (PreciseCutPoint::wedge),
/// or away from them if the triangles of another edge defined alike are in the angle between its own ones, so that the contours of two edges
/// whose sheets touch along the segment without crossing do not cross either, while those of crossing sheets do;
/// a crossing of two cuts is the point where the perturbed planes of the triangle and of the two cuts meet, so it is on the side
/// of every other plane that the perturbation puts it on, as a vertex is;
/// the sweep goes over the projection of the perturbed triangle on the plane of the two axes least aligned with its normal
/// (dropping the axis of the largest component of the normal, the first one of equal components, or the axis of the leading
/// term of the perturbed normal if the exact one is zero), meeting the points in the order of one linear functional,
/// which is nearly the first projected coordinate; the const members may be called concurrently (less from a parallel sort),
/// while reset and addPoint must not run together with any other member
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

    /// sets point v: a vertex of the triangle, or a crossing of one of its sides with another triangle, or a crossing of an edge
    /// of another triangle with this triangle (whose tri is then the triangle of reset), or a crossing of two cuts of this triangle
    /// by the planes of tri and tri2 (a valid tri2 makes it one, and its seg and wedge are ignored then);
    /// v indexes a vector of the points, resized to the largest v + 1, so the points take small local ids,
    /// not those of the mesh vertices; a crossing is inside the triangle it crosses, as \ref doTriangleSegmentIntersect finds them,
    /// which only Debug checks, since it takes several orient3d per point, and a crossing of two cuts is where the segments of the cuts
    /// in this triangle cross, of which Debug checks only that it is not outside this triangle;
    /// a segment in the plane of its triangle crosses it after the perturbation, at a point that tends to an end of the segment
    /// or to a point inside it, and two cuts parallel before the perturbation cross after it, at a point that tends to a point of both;
    /// setting a point again forgets the orders found for its old definition, but not a failure, and so does setting one with a wedge
    /// after points defined alike were compared, since it may change their shifts;
    /// a vertex point that is not a corner of the triangle, repeated ids, a segment with an end on a vertex of its own triangle,
    /// a segment off the sides with a triangle other than this one (as one with a single end at a corner), a segment that does not
    /// cross the plane of its triangle, two cuts by the same triangle (with three common vertex ids, while distinct triangles in one plane
    /// give two parallel cuts, which cross as said above), a cut along a side or by this triangle (by a triangle with two or three corners
    /// of this triangle), two cuts whose crossing is another point also after the perturbation (by triangles with a common corner of this
    /// triangle, or with two common vertices, whose edge they cross where it crosses this triangle), and two cuts that stay parallel
    /// after the perturbation or whose crossing goes to infinity as it vanishes are failures (see failed)
    MRMESH_API void addPoint( VertId v, const PreciseCutPoint & p );

    /// whether point p comes before point q along the sweep: a strict total order, unless reset or addPoint failed (see failed)
    [[nodiscard]] MRMESH_API bool less( VertId p, VertId q ) const;

    /// the signs of the projected coordinates of q - p found by expanding the perturbation alone, as the sweep sees them:
    /// a coordinate whose leading term has a larger degree than that of the other one is zero; this is slow,
    /// while less and ccwFromBehind decide most pairs by faster rules; nullopt if the points coincide after the perturbation too,
    /// as points defined alike do (only the shifts of their edges tell them apart)
    [[nodiscard]] MRMESH_API std::optional<Vector2i> directionByPerturbation( VertId p, VertId q ) const;

    /// whether triangle abc of distinct points is counterclockwise in the projection, for any three points: the first rule that applies
    /// decides, of the rounded coordinates (if their errors cannot change the sign), a side through two of the points, and, with the answers
    /// kept till reset since the sweep asks about the same triangles often, a plane line through two of them (the lines of ccwByLines),
    /// the exact positions, and the expansion of the perturbation; where several rules apply, they agree
    [[nodiscard]] MRMESH_API bool ccw( VertId a, VertId b, VertId c ) const;

    /// the same as ccw, decided by a line through two of the points known from their definitions, as the ends of every edge
    /// of the cut triangle are: one side of the triangle, or the plane of another triangle (by which the cut along the line goes,
    /// also from a vertex of both at selfBoolean's folds), or the plane vcd through a vertex v of the triangle and an edge cd
    /// of another one (for v and the crossing of cd with this triangle, at every vertex), or the plane through two edges
    /// with a common vertex (for their crossings with this triangle), which ccw needs only for points on one line before the perturbation;
    /// a crossing of two cuts is on both of them;
    /// the points on one side stay on one line after the perturbation, and they are taken
    /// as bulging inward a little, so three of them make a counterclockwise triangle if they come in the order
    /// from vertex i+1 to vertex i of side i, up to a cyclic shift; so are the points on a cut with a crossing of two cuts among them,
    /// as the pieces of a cut split at its crossings are, bulging to the left of the cut in the direction that the sweep goes along it;
    /// where the line has the third point too otherwise, two of the points are defined alike in valid input, and the shifts of their edges decide;
    /// with a crossing of two cuts among the points, the first-order term of the shifts decides before the bulge, since only the shifts
    /// move those two apart (a failure if that term is zero, as with the difference of the shifts along the cut also after the perturbation,
    /// and so for a third point that is not a crossing of two cuts too, if a crossing of two cuts is on the line);
    /// nullopt if no two of the points are on a known line
    [[nodiscard]] MRMESH_API std::optional<bool> ccwByLines( VertId a, VertId b, VertId c ) const;

    /// the same as ccw found by expanding the perturbation alone, which is slow, while ccw decides most triangles by faster rules;
    /// nullopt if the points are on one line after the perturbation too, as the points of one side are, and two points defined alike,
    /// or if the expansion reaches a stage whose degrees do not fit in int64, which only crossings of two cuts with many vertices
    /// can give (as three crossings with no common plane take 21 vertices)
    [[nodiscard]] MRMESH_API std::optional<bool> ccwByPerturbation( VertId a, VertId b, VertId c ) const;

    /// whether the triangle of a point far behind pivot against the sweep, b and pivot is counterclockwise, for distinct b and pivot
    [[nodiscard]] MRMESH_API bool ccwFromBehind( VertId b, VertId pivot ) const;

    /// whether the input broke a precondition that reset or addPoint checks, or a predicate met points it cannot tell apart: two points
    /// defined alike that the shifts of their edges do not tell apart (a vertex twice, crossings of a side, crossings without wedges or
    /// with the same one, or with the segment parallel to the triangle before the perturbation, as with a zero normal, or crossings of
    /// the same two cuts), three points defined alike with their shifts on one line, two points defined alike on a known line (see ccwByLines)
    /// with a crossing of two cuts, among the points of the triangle or not, with the difference of their shifts along the line also after
    /// the perturbation, three points off the sides on one line also after the perturbation and without a crossing of two cuts among them
    /// (as three crossings of one cut by edges, while valid input has two at most, apart from points defined alike), or points whose expansion
    /// of the perturbation reaches a stage whose degrees do not fit in int64 (see ccwByPerturbation); it stays set till reset; the caller checks
    /// it after the addPoint calls and calls no predicate on the point of a failed addPoint, since they take the points as valid (less may be
    /// cyclic then, and Debug asserts may fire), while predicates on the other points stay callable; a failure that a predicate finds leaves
    /// the predicates callable, with arbitrary answers that the caller discards, and less a strict total order (points defined alike compare
    /// alike with every other point, and by their ids with one another if their shifts do not tell them apart)
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

    /// the shift of a point defined alike with others; defined in the .cpp file
    struct Shift;

    void addCrossing_( VertId v, const PreciseCutPoint & p );
    bool isCrossing_( VertId v ) const;
    int compareExact_( VertId p, VertId q, bool yFirst ) const;
    int exactOrient_( VertId a, VertId b, VertId c ) const;
    bool ccwExact_( VertId a, VertId b, VertId c ) const;
    int orientSlow_( VertId a, VertId b, VertId c ) const;
    std::optional<bool> ccwBySides_( VertId a, VertId b, VertId c ) const;
    std::optional<bool> ccwByPlanes_( VertId a, VertId b, VertId c ) const;
    std::optional<bool> ccwByShifts_( VertId a, VertId b, VertId c, const Line * line ) const;
    std::optional<Vector2i> coincidentDirection_( VertId p, VertId q ) const;
    std::optional<Vector2i> directionFromSide_( VertId p, VertId q ) const;
    std::optional<Vector2i> directionAlongCut_( VertId p, VertId q ) const;
    std::optional<Vector2i> perturbedDirection_( VertId p, VertId q, bool alike ) const;
    std::optional<Vector2i> shiftDirection_( VertId p, VertId q ) const;
    std::optional<Shift> shift_( VertId p ) const;
    std::optional<Line> commonLine_( VertId p, VertId q, bool withVertex ) const;
    Vector2i lineDirection_( const Line & l ) const;
    bool along_( const Vector2i & dir, VertId p, VertId q ) const;
    int sideOfPlane_( const std::array<PreciseVertCoords, 3> & plane, VertId x ) const;
    int crossingSideOfPlane_( const std::array<PreciseVertCoords, 3> & plane, VertId x ) const;
    void fail_() const;

    std::array<PreciseVertCoords, 3> tri_;
    Vector3i64 normal_;
    int kx_ = 0, ky_ = 1; // the projection axes, in this order ccw around the normal
    std::array<Vector2i, 3> sideDirs_; // the signs of the projected coordinates of every side i, from vertex i to vertex i+1
    Vector<Vector2d, VertId> approx_; // the projected coordinates of every point, rounded, to decide most comparisons
    std::unique_ptr<Impl> impl_;
};

/// \}

} // namespace MR
