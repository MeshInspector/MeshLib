#pragma once

#include "MRId.h"
#include "MRVector2.h"
#include <cfloat>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

struct Polyline2ProjectionResult
{
    /// closest line id on polyline
    UndirectedEdgeId line;
    /// closest point on polyline, transformed by xf if it is given
    Vector2f point;
    /// squared distance from pt to proj
    float distSq = 0;
};
 
/**
 * \brief computes the closest point on two dimensional polyline to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
 * \param xf mesh-to-point transformation, if not specified then identity transformation is assumed
 */
MRMESH_API Polyline2ProjectionResult findProjectionOnPolyline2( const Vector2f& pt, const Polyline2& polyline,
    float upDistLimitSq = FLT_MAX, AffineXf2f* xf = nullptr );

struct Polyline2ProjectionWithOffsetResult
{
    /// closest line id on polyline
    UndirectedEdgeId line;
    /// closest point on polyline, transformed by xf if it is given
    Vector2f point;
    /// distance from offset point to proj
    float dist = 0;
};

/**
 * \brief computes the closest point on polyline to given point, respecting each edge offset
 * \param offsetPerEdge offsetPerEdge offset for each edge of polyline
 * \param upDistLimit upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimit and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 */
MRMESH_API Polyline2ProjectionWithOffsetResult findProjectionOnPolyline2WithOffset( const Vector2f& pt, const Polyline2& polyline,
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, float upDistLimit = FLT_MAX, AffineXf2f* xf = nullptr );

/// \}

} // namespace MR
