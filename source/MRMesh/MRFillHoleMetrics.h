#pragma once

#include "MRMeshMetrics.h"
#include "MRId.h"

namespace MR
{

/** \struct ComplexFillMetric
  * \ingroup FillHoleGroup
  *
  * This struct provides complex metric which fines new triangles for: \n
  * 1. Angle with neighbors:  (  (1-cos(x))/(1+cos(x))  )^4\n
  * 2. Triangle aspect ratio:   Rabc / (2 rabc)\n
  * 3. Triangle area (normalized by max loop edge length^2)\n
  * 
  * getTriangleMetric - grows with neighbors angles and triangle aspect ratio (R/2r)\n
  * getEdgeMetric - grows with angle
  * \sa \ref FillHoleMetric
  */
struct MRMESH_CLASS ComplexFillMetric final : FillHoleMetric
{
    // e is edge with left hole
    MRMESH_API ComplexFillMetric( const Mesh& mesh, EdgeId e );
    const VertCoords& points;
    float reverseCharacteristicTriArea;
    /// 1. Angle with neighbors:  (  (1-cos(x))/(1+cos(x))  )^4\n
    /// 2. Triangle aspect ratio:   Rabc / (2 rabc)\n
    /// 3. Triangle area (normalized by max loop edge length^2)\n
    MRMESH_API virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId& aOpposit, const VertId& cOpposit ) const override;

    /// Angle with neighbor:  (  (1-cos(x))/(1+cos(x))  )^4
    MRMESH_API virtual double getEdgeMetric( const VertId& a, const VertId& b, const VertId& left, const VertId& right ) const override;
};

/** \struct MRE::ParallelPlaneFillMetric
  * \ingroup FillHoleGroup
  *
  * This metric minimizes summary projection of new edges to plane normal, (try do produce edges parallel to plane)
  *
  * getTriangleMetric - ac projection to normal
  * getEdgeMetric - -ab projection to normal (it is needed to count last edge only once, as far as edge metric is called only for edge that connects parts of triangulation)
  * \sa \ref FillHoleMetric
  */
struct MRMESH_CLASS ParallelPlaneFillMetric final : FillHoleMetric
{
    /// If plane is not null use it's normal, otherwise find best plane for hole points
    MRMESH_API ParallelPlaneFillMetric( const Mesh& mesh, EdgeId e, const Plane3f* plane = nullptr );
    const VertCoords& points;
    Vector3f normal;
    /// Projection of ac to normal
    MRMESH_API virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId& aOpposit, const VertId& cOpposit ) const override;
    /// -projection of ab to normal (it is needed to count last edge only once, as far as edge metric is called only for edge that connects parts of triangulation)
    MRMESH_API virtual double getEdgeMetric( const VertId& a, const VertId& b, const VertId& left, const VertId& right ) const override;
};

} //namespace MR
