#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"

MR_EXTERN_C_BEGIN

/// is called for each triangle, if it is set
typedef double (*MRFillTriangleMetric)( MRVertId, MRVertId, MRVertId );
/// is called for each edge, if it is set
typedef double (*MRFillEdgeMetric)( MRVertId, MRVertId, MRVertId, MRVertId );
/// is called to combine metrics from different candidates, if it is not set it just summarizes input
typedef double (*MRFillCombineMetric)( double, double );

/** \struct MRFillHoleMetric
  * \brief Holds metrics for mrFillHole and mrBuildCylinderBetweenTwoHoles triangulation\n
  *
  * This is struct used as optimization metric of mrFillHole and mrBuildCylinderBetweenTwoHoles functions\n
  *
  * \sa \ref mrGetCircumscribedMetric
  * \sa \ref mrGetPlaneFillMetric
  * \sa \ref mrGetEdgeLengthFillMetric
  * \sa \ref mrGetEdgeLengthStitchMetric
  * \sa \ref mrGetComplexStitchMetric
  * \sa \ref mrFillHole
  * \sa \ref mrBuildCylinderBetweenTwoHoles
  */
typedef struct MRFillHoleMetric MRFillHoleMetric;

MRMESHC_API MRFillHoleMetric* mrFillHoleMetricNew( MRFillTriangleMetric triangleMetric, MRFillEdgeMetric edgeMetric, MRFillCombineMetric combineMetric );

MRMESHC_API void mrFillHoleMetricFree( MRFillHoleMetric* metric );

/// Computes combined metric after filling a hole
MRMESHC_API double mrCalcCombinedFillMetric( const MRMesh* mesh, const MRFaceBitSet* filledRegion, const MRFillHoleMetric* metric );

/// This metric minimizes the sum of circumcircle radii for all triangles in the triangulation.
/// It is rather fast to calculate, and it results in typically good triangulations.
MRMESHC_API MRFillHoleMetric* mrGetCircumscribedMetric( const MRMesh* mesh );

/// Same as mrGetCircumscribedFillMetric, but with extra penalty for the triangles having
/// normals looking in the opposite side of plane containing left of (e).
MRMESHC_API MRFillHoleMetric* mrGetPlaneFillMetric( const MRMesh* mesh, MREdgeId e );

/// Similar to mrGetPlaneFillMetric with extra penalty for the triangles having
/// normals looking in the opposite side of plane containing left of (e),
/// but the metric minimizes the sum of circumcircle radius times aspect ratio for all triangles in the triangulation.
MRMESHC_API MRFillHoleMetric* mrGetPlaneNormalizedFillMetric( const MRMesh* mesh, MREdgeId e );

/// This metric minimizes the sum of triangleMetric for all triangles in the triangulation
/// plus the sum edgeMetric for all edges inside and on the boundary of the triangulation.\n
/// Where\n
/// triangleMetric is proportional to weighted triangle area and triangle aspect ratio\n
/// edgeMetric grows with angle between triangles as ( ( 1 - cos( x ) ) / ( 1 + cos( x ) ) ) ^ 4.
MRMESHC_API MRFillHoleMetric* mrGetComplexFillMetric( const MRMesh* mesh, MREdgeId e );

/// This metric minimizes the maximal dihedral angle between the faces in the triangulation
/// and on its boundary, and it avoids creating too degenerate triangles;
///  for planar holes it is the same as getCircumscribedMetric
MRMESHC_API MRFillHoleMetric* mrGetUniversalMetric( const MRMesh* mesh );

/// This metric is for triangulation construction with minimal summed area of triangles.
/// Warning: this metric can produce degenerated triangles
MRMESHC_API MRFillHoleMetric* mrGetMinAreaMetric( const MRMesh* mesh );

MR_EXTERN_C_END
