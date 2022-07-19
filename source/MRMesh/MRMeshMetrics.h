#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRMesh.h"
#include <functional>


namespace MR
{
/// \addtogroup FillHoleGroup
/// \{

using FillTriangleMetric = std::function<double( VertId, VertId, VertId )>;
using FillEdgeMetric = std::function<double( VertId, VertId, VertId, VertId )>;
using FillCombineMetric = std::function<double( double, double )>;

/** \struct MR::FillHoleMetric
  * \brief Holds metrics for fillHole and buildCylinderBetweenTwoHoles triangulation\n
  * 
  * This is struct used as optimization metric of fillHole and buildCylinderBetweenTwoHoles functions\n
  * 
  * \sa \ref getCircumscribedFillMetric
  * \sa \ref getPlaneFillMetric
  * \sa \ref getEdgeLengthFillMetric
  * \sa \ref getEdgeLengthStitchMetric
  * \sa \ref getComplexStitchMetric
  * \sa \ref getCircumscribedStitchMetric
  * \sa \ref fillHole
  * \sa \ref buildCylinderBetweenTwoHoles
  */
struct FillHoleMetric
{
    /// is called for each triangle, if it is set
    FillTriangleMetric triangleMetric;
    /// is called for each edge, if it is set
    FillEdgeMetric edgeMetric;
    /// is called to combine metrics from different candidates, if it is not set it just summarizes input
    FillCombineMetric combineMetric;
};

/// Provides triangle metric as circumscribed circle diameter
MRMESH_API FillHoleMetric getCircumscribedFillMetric( const Mesh& mesh );

/// As far as hole is planar, only outside triangles should have penalty,\n
/// this metric is good for planar holes
/// 
/// Provides triangle metric as area
MRMESH_API FillHoleMetric getPlaneFillMetric( const Mesh& mesh, EdgeId e );

/// As far as hole is planar, only outside triangles should have penalty,\n
/// this metric is good for planar holes
/// 
/// Provides triangle metric as area
MRMESH_API FillHoleMetric getPlaneNormalizedFillMetric( const Mesh& mesh, EdgeId e );

/// Forbids connecting vertices from the same hole \n
/// Complex metric for non-trivial holes, forbids degenerate triangles\n
/// 
/// triangleMetric - grows with neighbors angles and triangle aspect ratio( R / 2r )\n
/// edgeMetric - grows with angle
MRMESH_API FillHoleMetric getComplexStitchMetric( const Mesh& mesh );

/// Simple metric minimizing edge length
MRMESH_API FillHoleMetric getEdgeLengthFillMetric( const Mesh& mesh );

/// Forbids connecting vertices from the same hole \n
/// Simple metric minimizing edge length
MRMESH_API FillHoleMetric getEdgeLengthStitchMetric( const Mesh& mesh );

/// Forbids connecting vertices from the same hole \n
/// Provides triangle metric as circumscribed circle diameter
MRMESH_API FillHoleMetric getCircumscribedStitchMetric( const Mesh& mesh );

/// Forbids connecting vertices from the same hole \n
/// All new faces should be parallel to given direction
MRMESH_API FillHoleMetric getVerticalStitchMetric( const Mesh& mesh, const Vector3f& upDir );

/// This struct provides complex metric which fines new triangles for: \n
/// 1. Angle with neighbors : ( ( 1 - cos( x ) ) / ( 1 + cos( x ) ) ) ^ 4\n
/// 2. Triangle aspect ratio : Rabc / ( 2 rabc )\n
/// 3. Triangle area( normalized by max loop edge length ^ 2 )\n
///
/// triangleMetric - grows with neighbors angles and triangle aspect ratio( R / 2r )\n
/// edgeMetric - grows with angle
MRMESH_API FillHoleMetric getComplexFillMetric( const Mesh& mesh, EdgeId e );

/// This metric minimizes summary projection of new edges to plane normal, (try do produce edges parallel to plane)
/// 
/// triangleMetric - ac projection to normal
/// edgeMetric - -ab projection to normal( it is needed to count last edge only once, as far as edge metric is called only for edge that connects parts of triangulation )
MRMESH_API FillHoleMetric getParallelPlaneFillMetric( const Mesh& mesh, EdgeId e, const Plane3f* plane = nullptr );

/// This metric minimizes maximum dihedral angle in triangulation
MRMESH_API FillHoleMetric getMaxDihedralAngleMetric( const Mesh& mesh );

/// \}

}