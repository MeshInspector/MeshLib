#pragma once

#include "MRVector2.h"
#include "MRId.h"
#include <array>
#include <functional>

namespace MR
{

/// \defgroup PrecisePredicates2
/// \ingroup MathGroup
/// \{

/// return true if the smallest rotation from vector (a) to vector (b) is in counter-clock-wise direction;
/// uses simulation-of-simplicity to avoid "vectors are collinear"
MRMESH_API bool ccw( const Vector2i & a, const Vector2i & b );

/// return true if the smallest rotation from vector (a-c) to vector (b-c) is in counter-clock-wise direction;
/// uses simulation-of-simplicity to avoid "vectors are collinear"
inline bool ccw( const Vector2i & a, const Vector2i & b, const Vector2i & c )
    { return ccw( a - c, b - c ); }

struct PreciseVertCoords2
{
    VertId id;   ///< unique id of the vertex (in both contours)
    Vector2i pt; ///< integer coordinates of the vertex
};

/// first sorts the indices in ascending order, then calls the predicate for sorted points
MRMESH_API bool ccw( const std::array<PreciseVertCoords2, 3> & vs );
MRMESH_API bool ccw( const PreciseVertCoords2* vs );

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

/// finds intersection precise, using high precision int inside
/// this function input should have intersection
[[nodiscard]] MRMESH_API Vector2f findSegmentSegmentIntersectionPrecise( 
    const Vector2f& a, const Vector2f& b, const Vector2f& c, const Vector2f& d,
    CoordinateConverters2 converters );

/// \}

} // namespace MR
