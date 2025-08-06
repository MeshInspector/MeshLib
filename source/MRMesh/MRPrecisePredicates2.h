#pragma once

#include "MRVector2.h"
#include "MRId.h"
#include <array>
#include <functional>

namespace MR
{

/// \ingroup MathGroup
/// \{

struct PreciseVertCoord
{
    VertId id;   ///< unique id of the vertex (in both contours)
    int pt;      ///< coordinate
};

/// return true if l is smaller than r,
/// uses simulation-of-simplicity (assuming larger perturbations of points with smaller id) to avoid "coordinates are the same"
[[nodiscard]] inline bool smaller( const PreciseVertCoord& l, const PreciseVertCoord& r )
    { return l.pt < r.pt || ( l.pt == r.pt && r.id < l.id ); }

/// return true if the smallest rotation from vector (a) to vector (b) is in counter-clock-wise direction;
/// uses simulation-of-simplicity (assuming perturbations a >> b) to avoid "vectors are collinear"
[[nodiscard]] MRMESH_API bool ccw( const Vector2i & a, const Vector2i & b );

/// return true if the smallest rotation from vector (a-c) to vector (b-c) is in counter-clock-wise direction;
/// uses simulation-of-simplicity (assuming perturbations a >> b >> c) to avoid "vectors are collinear"
[[nodiscard]] inline bool ccw( const Vector2i & a, const Vector2i & b, const Vector2i & c )
    { return ccw( a - c, b - c ); }

struct PreciseVertCoords2
{
    VertId id;   ///< unique id of the vertex (in both contours)
    Vector2i pt; ///< integer coordinates of the vertex
};

/// return true if the smallest rotation from vector (vs[0]-vs[2]) to vector (vs[1]-vs[2]) is in counter-clock-wise direction;
/// uses simulation-of-simplicity (assuming larger perturbations of points with smaller id) to avoid "vectors are collinear"
[[nodiscard]] MRMESH_API bool ccw( const std::array<PreciseVertCoords2, 3> & vs );
[[nodiscard]] MRMESH_API bool ccw( const PreciseVertCoords2* vs );

/// given the line passing via points vs[0] and vs[1], which defines linear signed scalar distance field:
/// zero on the line, positive for x where ccw(vs[0], vs[1], x) == true, and negative for x where ccw(vs[0], vs[1], x) == false
/// finds whether sdistance(vs[2]) < sdistance(vs[3]);
/// avoids equality of signed distances using simulation-of-simplicity approach (assuming larger perturbations of points with smaller id)
[[nodiscard]] MRMESH_API bool smaller2( const std::array<PreciseVertCoords2, 4> & vs );

/// considers 3D points obtained from 2D inputs by moving each point on paraboloid: z = x*x+y*y;
/// returns true if the plane with orientated triangle ABC has D point at the left;
/// uses simulation-of-simplicity to avoid "D is exactly on plane"
[[nodiscard]] MRMESH_API bool orientParaboloid3d( const Vector2i & a, const Vector2i & b, const Vector2i & c );
[[nodiscard]] inline bool orientParaboloid3d( const Vector2i & a, const Vector2i & b, const Vector2i & c, const Vector2i & d )
    { return orientParaboloid3d( a - d, b - d, c - d ); }

/// return true if 4th point in array lays inside circumcircle of first 3 points based triangle
[[nodiscard]] MRMESH_API bool inCircle( const std::array<PreciseVertCoords2, 4>& vs );
[[nodiscard]] MRMESH_API bool inCircle( const PreciseVertCoords2* vs );

struct SegmentSegmentIntersectResult
{
    bool doIntersect = false;   ///< whether the segments intersect
    bool cIsLeftFromAB = false; ///< whether the directed line AB has C point at the left

    explicit operator bool() const { return doIntersect; }
};

/// checks whether the segments AB (indices 01) and segments CD (indices 23) intersect;
/// uses simulation-of-simplicity to avoid edge-segment intersections and co-planarity
[[nodiscard]] MRMESH_API SegmentSegmentIntersectResult doSegmentSegmentIntersect(
    const std::array<PreciseVertCoords2, 4> & vs );

/// float-to-int coordinate converter
using ConvertToIntVector2 = std::function<Vector2i( const Vector2f& )>;
/// int-to-float coordinate converter
using ConvertToFloatVector2 = std::function<Vector2f( const Vector2i& )>;
/// this struct contains coordinate converters float-int-float
struct CoordinateConverters2
{
    ConvertToIntVector2 toInt{};
    ConvertToFloatVector2 toFloat{};
};

/// given line segment s=01 and two other segments sa=23, sb=45 known to intersect it, finds the order of intersection using precise predicates:
/// true:  s[0], s ^ sa, s ^ sb, s[1]
/// false: s[0], s ^ sb, s ^ sa, s[1]
/// segments sa and sb can have at most one shared point, all other points must be unique
[[nodiscard]] MRMESH_API bool segmentIntersectionOrder( const std::array<PreciseVertCoords2, 6> & vs );

/// finds intersection precise, using high precision int inside
/// this function input should have intersection
[[nodiscard]] MRMESH_API Vector2i findSegmentSegmentIntersectionPrecise(
    const Vector2i& a, const Vector2i& b, const Vector2i& c, const Vector2i& d );

/// finds intersection precise, using high precision int inside
/// this function input should have intersection
[[nodiscard]] MRMESH_API Vector2f findSegmentSegmentIntersectionPrecise(
    const Vector2f& a, const Vector2f& b, const Vector2f& c, const Vector2f& d,
    CoordinateConverters2 converters );

/// \}

} // namespace MR
