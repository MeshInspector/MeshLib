#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRMesh.h"

namespace MR
{
/// \addtogroup FillHoleGroup
/// \{

/** \struct MR::FillHoleMetric
  * \brief Provides interface for controlling fillHole and buildCylinderBetweenTwoHoles triangulation\n
  * 
  * This is abstract struct used as optimization metric of fillHole and buildCylinderBetweenTwoHoles functions\n
  * 
  * getTriangleMetric is called for each new triangle candidate and its two known neighbors\n
  * getEdgeMetric is called only once when two part of triangulation are merging (edge between two last triangles)
  * 
  * \sa \ref CircumscribedFillMetric
  * \sa \ref PlaneFillMetric
  * \sa \ref EdgeLengthFillMetric
  * \sa \ref ComplexStitchMetric
  * \sa \ref CircumscribedStitchMetric
  * \sa \ref EdgeLengthStitchMetric
  * \sa \ref fillHole
  * \sa \ref buildCylinderBetweenTwoHoles
  */
struct FillHoleMetric
{
    FillHoleMetric() = default;
    virtual ~FillHoleMetric() = default;
    virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId& aOpposit, const VertId& cOpposit ) const = 0;
    virtual double getEdgeMetric( const VertId& a, const VertId& b, const VertId& left, const VertId& right ) const = 0;
    /// if metric's `getEdgeMetric` returns constant this flag should be false
    bool hasEdgeMetric{ true };
};

/** \struct MR::CircumscribedFillMetric
  * Provides triangle metric as circumscribed circle diameter\n
  * getEdgeMetric - always returns zero
  * \sa \ref FillHoleMetric
  */
struct MRMESH_CLASS CircumscribedFillMetric final : FillHoleMetric
{
    CircumscribedFillMetric( const Mesh& mesh ) : points{mesh.points} { hasEdgeMetric = false; }
    const VertCoords& points;
    /// circumscribed circle diameter
    MRMESH_API virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const override;

    /// 0.0
    MRMESH_API virtual double getEdgeMetric( const VertId& a, const VertId& b, const VertId& left, const VertId& right ) const override;
};

/** \struct MR::PlaneFillMetric
  * As far as hole is planar, only outside triangles should have penalty,\n
  * this metric is good for planar holes 
  * 
  * Provides triangle metric as area 
  * 'getEdgeMetric' - always returns zero
  * \sa \ref FillHoleMetric
  */
struct MRMESH_CLASS PlaneFillMetric final : FillHoleMetric
{
    /// e is edge with left hole, it is needed to find normal of hole plane
    MRMESH_API PlaneFillMetric( const Mesh& mesh, EdgeId e );
    const VertCoords& points;
    Vector3d norm;
    /// DBL_MAX if normal direction is different of hole plane norm, otherwise circumscribed circle diameter
    MRMESH_API virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const override;

    /// 0.0
    MRMESH_API virtual double getEdgeMetric( const VertId& a, const VertId& b, const VertId& left, const VertId& right ) const override;
};

/** \struct MR::PlaneNormalizedFillMetricPlaneNormalizedFillMetric
  * As far as hole is planar, only outside triangles should have penalty,\n
  * this metric is good for planar holes
  *
  * Provides triangle metric as area and triangle * aspect ratio
  * 'getEdgeMetric' - always returns zero
  * \sa \ref PlaneFillMetric
  * \sa \ref FillHoleMetric
  */
struct MRMESH_CLASS PlaneNormalizedFillMetric final : FillHoleMetric
{
    /// e is edge with left hole, it is needed to find normal of hole plane
    MRMESH_API PlaneNormalizedFillMetric( const Mesh & mesh, EdgeId e );
    const VertCoords& points;
    Vector3d norm;
    /// DBL_MAX if normal direction is different of hole plane norm, otherwise circumscribed circle diameter * aspect ratio
    MRMESH_API virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const override;
    /// 0.0
    MRMESH_API virtual double getEdgeMetric( const VertId& a, const VertId& b, const VertId& left, const VertId& right ) const override;
};

/** \struct MR::ComplexStitchMetric
  * Forbids connecting vertices from different holes \n
  * 
  * Complex metric for non-trivial holes, forbids degenerate triangles\n
  *
  * getTriangleMetric - grows with neighbors angles and triangle aspect ratio (R/2r)\n
  * getEdgeMetric - grows with angle
  * \sa \ref FillHoleMetric
  */
struct MRMESH_CLASS ComplexStitchMetric final : FillHoleMetric
{
    /// a,b are edges with left holes
    MRMESH_API ComplexStitchMetric( const Mesh& mesh );
    const VertCoords& points;
    /// Sum of edge metric for abcc' bcaa' + circumcircleDiameter(a,b,c)
    MRMESH_API virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId& aOpposit, const VertId& cOpposit ) const override;
    /// Grows with ab angle
    MRMESH_API virtual double getEdgeMetric( const VertId& a, const VertId& b, const VertId& left, const VertId& right ) const override;
};

/** \struct MR::EdgeLengthFillMetric
  * Simple metric minimizing edge length
  * \sa \ref FillHoleMetric
  */
struct MRMESH_CLASS EdgeLengthFillMetric final : FillHoleMetric
{
    EdgeLengthFillMetric( const Mesh& mesh ) : points{ mesh.points }
    {
    };
    const VertCoords& points;
    MRMESH_API virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const override;
    MRMESH_API virtual double getEdgeMetric( const VertId& a, const VertId& b, const VertId&, const VertId& ) const override;
};

/** \struct MR::EdgeLengthStitchMetric
  * Forbids connecting vertices from different holes\n
  *
  * Simple metric minimizing edge length
  * \sa \ref FillHoleMetric
  */
struct MRMESH_CLASS EdgeLengthStitchMetric final : FillHoleMetric
{
    /// a,b are edges with left holes
    MRMESH_API EdgeLengthStitchMetric( const Mesh& mesh );
    const VertCoords& points;
    MRMESH_API virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const override;
    ///  0.0
    MRMESH_API virtual double getEdgeMetric( const VertId& a, const VertId& b, const VertId&, const VertId& ) const override;
};

/** \struct MR::CircumscribedStitchMetric
  * Forbids connecting vertices from different holes\n
  * 
  * Provides triangle metric as circumscribed circle diameter\n
  * getEdgeMetric - always returns zero
  * \sa \ref FillHoleMetric
  */
struct MRMESH_CLASS CircumscribedStitchMetric final : FillHoleMetric
{
    MRMESH_API CircumscribedStitchMetric( const Mesh& mesh );
    const VertCoords& points;
    /// Circumscribed circle diameter
    MRMESH_API virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const override;
    /// 0.0
    MRMESH_API virtual double getEdgeMetric( const VertId&, const VertId&, const VertId&, const VertId& ) const override;
};

/** \struct MR::VerticalStitchMetric
  * Forbids connecting vertices from different holes\n
  *
  * All new faces should be parallel to given direction\n
  * getEdgeMetric - always returns zero
  * \sa \ref FillHoleMetric
  */
struct MRMESH_CLASS VerticalStitchMetric final : FillHoleMetric
{
    MRMESH_API VerticalStitchMetric( const Mesh& mesh, const Vector3f& upDir );
    const VertCoords& points;
    Vector3f upDirection;

    /// Fines for non parallel to upDir
    MRMESH_API virtual double getTriangleMetric( const VertId& a, const VertId& b, const VertId& c, const VertId&, const VertId& ) const override;
    /// 0.0
    MRMESH_API virtual double getEdgeMetric( const VertId&, const VertId&, const VertId&, const VertId& ) const override;
};

/// \}

}