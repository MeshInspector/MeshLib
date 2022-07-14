#pragma once
#include "MRId.h"
#include "MRVector3.h"
#include <cfloat>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

struct PolylineProjectionResult
{
    /// closest line id on polyline
    UndirectedEdgeId line;
    /// closest point on polyline, transformed by xf if it is given
    Vector3f point;
    /// squared distance from pt to proj
    float distSq = 0;
};

/**
 * \brief computes the closest point on polyline to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 */
MRMESH_API PolylineProjectionResult findProjectionOnPolyline( const Vector3f& pt, const Polyline3& polyline,
    float upDistLimitSq = FLT_MAX, AffineXf3f* xf = nullptr );

struct PolylineProjectionWithOffsetResult
{
    /// closest line id on polyline
    UndirectedEdgeId line;
    /// closest point on polyline, transformed by xf if it is given
    Vector3f point;
    /// distance from offset point to proj
    float dist = 0;
};

/**
 * \brief computes the closest point on polyline to given point, respecting each edge offset
 * \param offsetPerEdge offset for each edge of polyline
 * \param upDistLimit upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimit and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 */
MRMESH_API PolylineProjectionWithOffsetResult findProjectionOnPolylineWithOffset( const Vector3f& pt, const Polyline3& polyline,
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, float upDistLimit = FLT_MAX, AffineXf3f* xf = nullptr );

/**
 * \brief computes the closest point on the mesh edges (specified by the tree) to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 */
MRMESH_API PolylineProjectionResult findProjectionOnMeshEdges( const Vector3f& pt, const Mesh& mesh, const AABBTreePolyline3& tree,
    float upDistLimitSq = FLT_MAX, AffineXf3f* xf = nullptr );

/// \}

} // namespace MR
