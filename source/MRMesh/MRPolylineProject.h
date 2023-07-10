#pragma once
#include "MRId.h"
#include "MRVector3.h"
#include <cfloat>
#include <functional>

namespace MR
{

/// \addtogroup AABBTreeGroup
/// \{

template<typename V>
struct PolylineProjectionResult
{
    /// closest line id on polyline
    UndirectedEdgeId line;
    /// closest point on polyline, transformed by xf if it is given
    V point;
    /// squared distance from pt to proj
    float distSq = 0;
};

/**
 * \brief computes the closest point on polyline to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 */
MRMESH_API PolylineProjectionResult2 findProjectionOnPolyline2( const Vector2f& pt, const Polyline2& polyline,
    float upDistLimitSq = FLT_MAX, AffineXf2f* xf = nullptr, float loDistLimitSq = 0 );

/**
 * \brief computes the closest point on polyline to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 */
MRMESH_API PolylineProjectionResult3 findProjectionOnPolyline( const Vector3f& pt, const Polyline3& polyline,
    float upDistLimitSq = FLT_MAX, AffineXf3f* xf = nullptr, float loDistLimitSq = 0 );

/**
 * \brief computes the closest point on polyline to given straight line
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 */
MRMESH_API PolylineProjectionResult3 findProjectionOnPolyline( const Line3f& ln, const Polyline3& polyline,
    float upDistLimitSq = FLT_MAX, AffineXf3f* xf = nullptr, float loDistLimitSq = 0 );

template<typename V>
struct PolylineProjectionWithOffsetResult
{
    /// closest line id on polyline
    UndirectedEdgeId line;
    /// closest point on polyline, transformed by xf if it is given
    V point;
    /// distance from offset point to proj
    float dist = 0;
};

/**
 * \brief computes the closest point on polyline to given point, respecting each edge offset
 * \param offsetPerEdge offset for each edge of polyline
 * \param upDistLimit upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimit and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimit low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 */
MRMESH_API Polyline2ProjectionWithOffsetResult findProjectionOnPolyline2WithOffset( const Vector2f& pt, const Polyline2& polyline,
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, float upDistLimit = FLT_MAX, AffineXf2f* xf = nullptr, float loDistLimit = 0 );

/**
 * \brief computes the closest point on polyline to given point, respecting each edge offset
 * \param offsetPerEdge offset for each edge of polyline
 * \param upDistLimit upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimit and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimit low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 */
MRMESH_API PolylineProjectionWithOffsetResult3 findProjectionOnPolylineWithOffset( const Vector3f& pt, const Polyline3& polyline,
    const Vector<float, UndirectedEdgeId>& offsetPerEdge, float upDistLimit = FLT_MAX, AffineXf3f* xf = nullptr, float loDistLimit = 0 );

template<typename V>
using FoundEdgeCallback = std::function<void( UndirectedEdgeId, const V& closestPt, float distSq )>;
using FoundEdgeCallback2 = FoundEdgeCallback<Vector2f>;
using FoundEdgeCallback3 = FoundEdgeCallback<Vector3f>;

/// Finds all edges of given polyline that cross or touch given ball (center, radius)
MRMESH_API void findEdgesInBall( const Polyline2& polyline, const Vector2f& center, float radius, const FoundEdgeCallback2& foundCallback, AffineXf2f* xf = nullptr );

/// Finds all edges of given polyline that cross or touch given ball (center, radius)
MRMESH_API void findEdgesInBall( const Polyline3& polyline, const Vector3f& center, float radius, const FoundEdgeCallback3& foundCallback, AffineXf3f* xf = nullptr );

/// Finds all edges of given mesh edges (specified by the tree) that cross or touch given ball (center, radius)
MRMESH_API void findMeshEdgesInBall( const Mesh& mesh, const AABBTreePolyline3& tree, const Vector3f& center, float radius, const FoundEdgeCallback3& foundCallback, AffineXf3f* xf = nullptr );

/**
 * \brief computes the closest point on the mesh edges (specified by the tree) to given point
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 */
MRMESH_API PolylineProjectionResult3 findProjectionOnMeshEdges( const Vector3f& pt, const Mesh& mesh, const AABBTreePolyline3& tree,
    float upDistLimitSq = FLT_MAX, AffineXf3f* xf = nullptr, float loDistLimitSq = 0 );

/**
 * \brief computes the closest point on the mesh edges (specified by the tree) to given straight line
 * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid point
 * \param xf polyline-to-point transformation, if not specified then identity transformation is assumed
 * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
 */
MRMESH_API PolylineProjectionResult3 findProjectionOnMeshEdges( const Line3f& ln, const Mesh& mesh, const AABBTreePolyline3& tree,
    float upDistLimitSq = FLT_MAX, AffineXf3f* xf = nullptr, float loDistLimitSq = 0 );

/// \}

} // namespace MR
