#pragma once
#include "exports.h"
#include "MRMesh/MRMeshMetrics.h"
#include "MRMesh/MRId.h"

namespace MRE
{

/** \struct MRE::ComplexFillMetric
  * \ingroup FillHoleGroup
  *
  * This struct provides complex metric which fines new triangles for: \n
  * 1. Angle with neighbors:  (  (1-cos(x))/(1+cos(x))  )^4\n
  * 2. Triangle aspect ratio:   Rabc / (2 rabc)\n
  * 3. Triangle area (normalized by max loop edge length^2)\n
  * 
  * getTriangleMetric - grows with neighbors angles and triangle aspect ratio (R/2r)\n
  * getEdgeMetric - grows with angle
  * \sa \ref MR::FillHoleMetric
  */
struct MREALGORITHMS_CLASS ComplexFillMetric final : MR::FillHoleMetric
{
    // e is edge with left hole
    MREALGORITHMS_API ComplexFillMetric( const MR::Mesh& mesh, MR::EdgeId e );
    const MR::VertCoords& points;
    float reverseCharacteristicTriArea;
    /// 1. Angle with neighbors:  (  (1-cos(x))/(1+cos(x))  )^4\n
    /// 2. Triangle aspect ratio:   Rabc / (2 rabc)\n
    /// 3. Triangle area (normalized by max loop edge length^2)\n
    MREALGORITHMS_API virtual double getTriangleMetric( const MR::VertId& a, const MR::VertId& b, const MR::VertId& c, const MR::VertId& aOpposit, const MR::VertId& cOpposit ) const override;

    /// Angle with neighbor:  (  (1-cos(x))/(1+cos(x))  )^4
    MREALGORITHMS_API virtual double getEdgeMetric( const MR::VertId& a, const MR::VertId& b, const MR::VertId& left, const MR::VertId& right ) const override;
};

/** \struct MRE::ParallelPlaneFillMetric
  * \ingroup FillHoleGroup
  *
  * This metric minimizes summary projection of new edges to plane normal, (try do produce edges parallel to plane)
  *
  * getTriangleMetric - ac projection to normal
  * getEdgeMetric - -ab projection to normal (it is needed to count last edge only once, as far as edge metric is called only for edge that connects parts of triangulation)
  * \sa \ref MR::FillHoleMetric
  */
struct MREALGORITHMS_CLASS ParallelPlaneFillMetric final : MR::FillHoleMetric
{
    /// If plane is not null use it's normal, otherwise find best plane for hole points
    MREALGORITHMS_API ParallelPlaneFillMetric( const MR::Mesh& mesh, MR::EdgeId e, const MR::Plane3f* plane = nullptr );
    const MR::VertCoords& points;
    MR::Vector3f normal;
    /// Projection of ac to normal
    MREALGORITHMS_API virtual double getTriangleMetric( const MR::VertId& a, const MR::VertId& b, const MR::VertId& c, const MR::VertId& aOpposit, const MR::VertId& cOpposit ) const override;
    /// -projection of ab to normal (it is needed to count last edge only once, as far as edge metric is called only for edge that connects parts of triangulation)
    MREALGORITHMS_API virtual double getEdgeMetric( const MR::VertId& a, const MR::VertId& b, const MR::VertId& left, const MR::VertId& right ) const override;
};

}