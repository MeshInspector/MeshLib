#pragma once

#include "MRVector3.h"
#include "MRId.h"
#include <array>
#include <optional>

namespace MR
{

/// \ingroup MathGroup
/// \{

/// Precise orientation predicate for the origin against the plane of triangle ABC.
/// Returns true iff dot( a, cross( b, c ) ) > 0, i.e. (a, b, c) form a right-handed vector triple /
/// the tetrahedron (origin, a, b, c) has positive signed volume. Geometrically the origin then lies
/// on the side that the right-hand normal cross( b - a, c - a ) of triangle ABC points away from
/// (equivalently, seen from the origin A->B->C winds clockwise).
/// Simulation-of-simplicity resolves the degenerate case dot( a, cross( b, c ) ) == 0 (origin exactly
/// on plane ABC) deterministically, so the predicate is never "undefined".
MRMESH_API bool orient3d( const Vector3i & a, const Vector3i & b, const Vector3i & c );

/// Precise orientation predicate for point D against the plane of triangle ABC
/// (the general case of the 3-argument overload above, which tests the origin instead of D).
/// Returns true iff dot( a - d, cross( b - d, c - d ) ) > 0, i.e. the tetrahedron (d, a, b, c) has
/// positive signed volume. Geometrically D then lies on the side that the right-hand normal
/// cross( b - a, c - a ) of triangle ABC points away from (equivalently, seen from D the vertices
/// A->B->C wind clockwise). Simulation-of-simplicity resolves the degenerate case (D exactly on
/// plane ABC) deterministically, so the predicate is never "undefined".
inline bool orient3d( const Vector3i & a, const Vector3i & b, const Vector3i & c, const Vector3i & d )
    { return orient3d( a - d, b - d, c - d ); }

struct PreciseVertCoords
{
    VertId id;   ///< unique id of the vertex (in both meshes)
    Vector3i pt; ///< integer coordinates of the vertex
};

/// Same predicate as orient3d( a, b, c, d ) evaluated on vs[0..3].pt, but first sorts the four
/// vertices by their id (ascending, flipping the result once per swap). The simulation-of-simplicity
/// perturbation depends on vertex id, so sorting makes the result depend only on the set of four
/// vertices and not on the order they are passed in: every call involving the same four vertices
/// agrees, which keeps orientation decisions consistent mesh-wide.
MRMESH_API bool orient3d( const std::array<PreciseVertCoords, 4> & vs );
MRMESH_API bool orient3d( const PreciseVertCoords* vs );

/// returns true if the points vs[2], vs[3], vs[4] are in counter-clockwise order on the plane
/// orthogonal to the direction vs[1]-vs[0], as seen by the viewer this direction points at,
/// i.e. if dot( vs[1]-vs[0], cross( vs[3]-vs[2], vs[4]-vs[2] ) ) is positive;
/// uses simulation-of-simplicity (assuming larger perturbations of points with smaller id)
/// to avoid "the direction is exactly parallel to the plane of the three points";
/// all five ids must be distinct, while the coordinates of the points can coincide arbitrarily
[[nodiscard]] MRMESH_API bool ccw3d( const std::array<PreciseVertCoords, 5> & vs );
[[nodiscard]] MRMESH_API bool ccw3d( const PreciseVertCoords* vs );

struct TriangleSegmentIntersectResult
{
    bool doIntersect = false;    ///< whether triangle and segment intersect
    bool dIsLeftFromABC = false; ///< value of orient3d( A, B, C, D ) (see there); "left" means the side the right-hand normal cross( B - A, C - A ) of triangle ABC points away from

    explicit operator bool() const { return doIntersect; }
};

/// checks whether triangle ABC (vertices 0,1,2) and segment DE (vertices 3,4) intersect.
/// Simulation-of-simplicity removes the degenerate cases (segment passing exactly through an edge
/// or vertex of the triangle, or lying in the triangle's plane), so the answer is always a clean yes/no.
[[nodiscard]] MRMESH_API TriangleSegmentIntersectResult doTriangleSegmentIntersect(
    const std::array<PreciseVertCoords, 5> & vs );

/// given line segment s (vertices 0,1) and two triangles ta (vertices 2,3,4) and tb (vertices 5,6,7),
/// each known to be pierced by s, returns the order in which s meets them, using precise predicates.
/// Here s^t denotes the point where segment s crosses triangle t, and s[0]/s[1] are the segment endpoints.
/// Walking s from s[0] to s[1]:
/// true:  order is s[0], s^ta, s^tb, s[1]  (ta met before tb)
/// false: order is s[0], s^tb, s^ta, s[1]  (tb met before ta)
/// triangles ta and tb may share at most two vertices; all remaining vertices must be unique
[[nodiscard]] MRMESH_API bool segmentIntersectionOrder( const std::array<PreciseVertCoords, 8> & vs );

/// translate then scale float-to-int coordinate converter
struct ConvertToIntVector
{
    Vector3d center;
    double invRange = 1;

    /// applies scaling only (without translation) with the following rounding to the nearest int
    [[nodiscard]] int scaleOnly( double v ) const
    {
        return (int)std::round( v * invRange );
    }

    /// double-to-int coordinates converter
    [[nodiscard]] Vector3i operator()( const Vector3d& p ) const
    {
        const auto d = p - center;
        return { scaleOnly( d.x ), scaleOnly( d.y ), scaleOnly( d.z ) };
    }

    /// float-to-int coordinates converter
    [[nodiscard]] Vector3i operator()( const Vector3f& p ) const
    {
        return operator()( Vector3d{ p } );
    }
};

/// scale then translate int-to-float coordinate converter
struct ConvertToFloatVector
{
    double range = 1;
    Vector3d center;

    /// int-to-double coordinates converter
    [[nodiscard]] Vector3d convert( const Vector3i& v ) const
    {
        return Vector3d{ v } * range + center;
    }

    /// int-to-float coordinates converter
    [[nodiscard]] Vector3f operator()( const Vector3i& v ) const
    {
        return Vector3f( convert( v ) );
    }
};

/// this struct contains coordinate converters float-int-float
struct CoordinateConverters
{
    ConvertToIntVector toInt;
    ConvertToFloatVector toFloat;
};

/// creates converter from Vector3f to Vector3i in Box range (int diapason is mapped to box range)
MRMESH_API ConvertToIntVector getToIntConverter( const Box3d& box );
/// creates converter from Vector3i to Vector3f in Box range (int diapason is mapped to box range)
MRMESH_API ConvertToFloatVector getToFloatConverter( const Box3d& box );

/// converts given points into integer coordinates in parallel
/// \param valid if given then only valid points are converted, and the content of other elements in the returned vector is undefined
[[nodiscard]] MRMESH_API Vector<Vector3i, VertId> computeIntCoords( const ConvertToIntVector& conv,
    const VertCoords& points, const VertBitSet* valid = nullptr );

/// converts given integer coordinates into float points in parallel
/// \param valid if given then only valid coordinates are converted, and the content of other elements in the returned vector is undefined
[[nodiscard]] MRMESH_API VertCoords computeFloatCoords( const ConvertToFloatVector& conv,
    const Vector<Vector3i, VertId>& intCoords, const VertBitSet* valid = nullptr );

/// given two line segments AB and CD located in one plane,
/// finds whether they intersect and if yes, computes their common point using integer-only arithmetic
[[nodiscard]] MRMESH_API std::optional<Vector3i> findTwoSegmentsIntersection( const Vector3i& ai, const Vector3i& bi, const Vector3i& ci, const Vector3i& di );

/// finds intersection precise, using high precision int inside
/// this function input should have intersection
[[nodiscard]] MRMESH_API Vector3f findTriangleSegmentIntersectionPrecise( 
    const Vector3f& a, const Vector3f& b, const Vector3f& c,
    const Vector3f& d, const Vector3f& e, 
    CoordinateConverters converters );

/// \}

}
