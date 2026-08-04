#pragma once

#include "MRVector3.h"
#include "MRId.h"
#include <array>
#include <optional>

namespace MR
{

/// \ingroup MathGroup
/// \{

/// returns true if the plane with orientated triangle ABC has 0 point at the left;
/// uses simulation-of-simplicity to avoid "0 is exactly on plane"
MRMESH_API bool orient3d( const Vector3i & a, const Vector3i & b, const Vector3i & c );

/// returns true if the plane with orientated triangle ABC has D point at the left;
/// uses simulation-of-simplicity to avoid "D is exactly on plane"
inline bool orient3d( const Vector3i & a, const Vector3i & b, const Vector3i & c, const Vector3i & d )
    { return orient3d( a - d, b - d, c - d ); }

struct PreciseVertCoords
{
    VertId id;   ///< unique id of the vertex (in both meshes)
    Vector3i pt; ///< integer coordinates of the vertex
};

/// first sorts the indices in ascending order, then calls the predicate for sorted points
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
    bool dIsLeftFromABC = false; ///< whether the plane with orientated triangle ABC has D point at the left

    explicit operator bool() const { return doIntersect; }
};

/// checks whether triangle ABC (indices 012) and segment DE (indices 34) intersect
/// uses simulation-of-simplicity to avoid edge-segment intersections and co-planarity
[[nodiscard]] MRMESH_API TriangleSegmentIntersectResult doTriangleSegmentIntersect(
    const std::array<PreciseVertCoords, 5> & vs );

/// given line segment s=01 and two triangles ta=234, tb=567 known to intersect it, finds the order of intersection using precise predicates:
/// true:  s[0], s ^ ta, s ^ tb, s[1]
/// false: s[0], s ^ tb, s ^ ta, s[1]
/// triangles ta and tb can have at most two shared points, all other points must be unique
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
