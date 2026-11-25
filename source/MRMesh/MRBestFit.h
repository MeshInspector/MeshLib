#pragma once

#include "MRSymMatrix3.h"
#include "MRPlane3.h"
#include "MRLine3.h"
#include "MRPolyline.h"
#include "MRAffineXf3.h"
#include "MRPointCloudPart.h"
#include "MRSphere.h"
#include "MRCylinder3.h"
#include "MRCone3.h"
#include "MRTorus3.h"
#include <array>
#include <optional>

namespace MR
{

/// \defgroup BestFitGroup BestFit
/// \ingroup MathGroup
/// \{

/// Class to accumulate points and make best line / plane approximation
class PointAccumulator
{
public:
    MRMESH_API void addPoint( const Vector3d & pt );
    MRMESH_API void addPoint( const Vector3d & pt, double weight );
    void addPoint( const Vector3f& pt ) { addPoint( Vector3d( pt ) ); }
    void addPoint( const Vector3f& pt, float weight ) { addPoint( Vector3d( pt ), weight ); }

    /// computes the best approximating plane from the accumulated points
    MRMESH_API Plane3d getBestPlane() const;
    Plane3f getBestPlanef() const { return Plane3f( getBestPlane() ); }
    /// computes the best approximating line from the accumulated points
    MRMESH_API Line3d getBestLine() const;
    Line3f getBestLinef() const { return Line3f( getBestLine() ); }

    /// computes centroid and eigenvectors/eigenvalues of the covariance matrix of the accumulated points
    MRMESH_API bool getCenteredCovarianceEigen( Vector3d & centroid, Matrix3d & eigenvectors, Vector3d & eigenvalues ) const;
    MRMESH_API bool getCenteredCovarianceEigen( Vector3f& centroid, Matrix3f& eigenvectors, Vector3f& eigenvalues ) const;

    /// returns the transformation that maps (0,0,0) into point centroid,
    /// and maps vectors (1,0,0), (0,1,0), (0,0,1) into first, second, third eigenvectors corresponding to ascending eigenvalues
    MRMESH_API AffineXf3d getBasicXf() const;
    AffineXf3f getBasicXf3f() const { return AffineXf3f( getBasicXf() ); }

    /// returns 4 transformations, each maps (0,0,0) into point centroid,
    /// and maps vectors (1,0,0), (0,1,0), (0,0,1) into +/- first, +/- second, +/- third eigenvectors (forming positive reference frame) corresponding to ascending eigenvalues
    MRMESH_API std::array<AffineXf3d, 4> get4BasicXfs() const;
    MRMESH_API std::array<AffineXf3f, 4> get4BasicXfs3f() const;

    bool valid() const { return sumWeight_ > 0; };

private:
    double sumWeight_ = 0;
    Vector3d momentum1_;
    SymMatrix3d momentum2_;
};

/// Adds in existing PointAccumulator all given points
MRMESH_API void accumulatePoints( PointAccumulator& accum, const std::vector<Vector3f>& points, const AffineXf3f* xf = nullptr );

/// Adds in existing PointAccumulator all given weighed points
MRMESH_API void accumulateWeighedPoints( PointAccumulator& accum, const std::vector<Vector3f>& points, const std::vector<float>& weights, const AffineXf3f* xf = nullptr );

/// Adds in existing PointAccumulator all mesh face centers with the weight equal to face area
MRMESH_API void accumulateFaceCenters( PointAccumulator& accum, const MeshPart& mp, const AffineXf3f* xf = nullptr );

/// Adds in existing PointAccumulator all line centers with the weight equal to the length line
MRMESH_API void accumulateLineCenters( PointAccumulator& accum, const Polyline3& pl, const AffineXf3f* xf = nullptr );

/// Adds in existing PointAccumulator all points from the cloud (region) with weight 1
MRMESH_API void accumulatePoints( PointAccumulator& accum, const PointCloudPart& pcp, const AffineXf3f* xf = nullptr );

/// \}

/// Class to accumulate planes to find then their crossing point
class PlaneAccumulator
{
public:
    MRMESH_API void addPlane( const Plane3d & pl );
    void addPlane( const Plane3f& pl ) { addPlane( Plane3d( pl ) ); }

    /// computes the point that minimizes the sum of squared distances to accumulated planes;
    /// if such point is not unique then returns the one closest to p0
    /// \param tol relative epsilon-tolerance for too small number detection
    /// \param rank optional output for solution matrix rank according to given tolerance
    /// \param space rank=1: unit normal to solution plane, rank=2: unit direction of solution line, rank=3: zero vector
    MRMESH_API Vector3d findBestCrossPoint( const Vector3d & p0, double tol, int * rank = nullptr, Vector3d * space = nullptr ) const;
    MRMESH_API Vector3f findBestCrossPoint( const Vector3f & p0, float tol, int * rank = nullptr, Vector3f * space = nullptr ) const;

private:
    SymMatrix3d mat_;
    Vector3d rhs_;
};

/// least squares sphere through the points (algebraic fit of the centered points): exact for the points of a sphere,
/// but with noisy points of a small cap its radius is biased low (e.g. by 16% if the standard deviation of the noise is 13% of the cap height);
/// for coplanar points the result is not unique, and the one with the center closest to their centroid is returned (the minimum-norm solution):
/// e.g. the sphere through the points of a circle is centered at that circle; returns nullopt for no points or coincident ones
[[nodiscard]] MRMESH_API std::optional<Sphere3d> findBestSphere( const std::vector<Vector3d>& points );

/// least squares circle through the points (algebraic fit of the centered points), stable for short arcs: exact for the points of a circle,
/// but with noisy points of a short arc its radius is biased low (e.g. by 13% if the standard deviation of the noise is 13% of the arc height);
/// for collinear points the result is not unique, and the one with the center closest to their centroid is returned (the minimum-norm solution):
/// its center is on their line, e.g. at their centroid for symmetric points; returns nullopt for no points or coincident ones
[[nodiscard]] MRMESH_API std::optional<Sphere2d> findBestCircle( const std::vector<Vector2d>& points );

/// finds the line crossing all given lines in the least squares sense,
/// e.g. the axis of a surface of revolution from the lines of its normals (lines with unit directions have equal weights);
/// returns nullopt if the line is not unique: e.g. all lines through the center of a sphere cross its normals,
/// and any line along the normal of a plane crosses its normals; with noisy normals of such surfaces, one of these lines can be returned
[[nodiscard]] MRMESH_API std::optional<Line3d> findRevolutionAxis( const std::vector<Line3d>& normalLines );

/// cylinder with the axis orthogonal to all given surface normals, and the radius fitted to the points by findBestCircle (biased low for noisy points of a short arc);
/// the center and the length of the cylinder span the projections of the points on the axis;
/// returns nullopt if findBestCircle does for the projections of the points, or if the normals do not define the axis:
/// the two smallest eigenvalues of the sum of their outer squares differ by less than 1e-6 of the largest one, e.g. for the normals of a plane
/// or of a sphere cap symmetric around its center; other sphere patches and noisy normals of a plane give a cylinder, so check its distances to the points
[[nodiscard]] MRMESH_API std::optional<Cylinder3d> findBestCylinderByNormals( const std::vector<Vector3d>& points, const std::vector<Vector3d>& normals );

/// cone with given axis fitted to the points: the line of the distance to the axis against the position along it, fitted orthogonally;
/// the height of the cone spans the points; returns nullopt if the line is nearly parallel (a cylinder) or orthogonal (a plane) to the axis
[[nodiscard]] MRMESH_API std::optional<Cone3d> findBestConeWithAxis( const std::vector<Vector3d>& points, const Line3d& axis );

/// cone tangent to all given planes (e.g. of mesh triangles) with consistently oriented normals, and the angle fitted to the points;
/// unlike findRevolutionAxis + findBestConeWithAxis, it works for a single strip of triangles; the planes with zero normals are ignored;
/// the height of the cone spans the points; returns nullopt if the planes do not define the cone: fewer than 3 distinct normals or the normals
/// within a few degrees of a revolution (the two smallest eigenvalues of their covariance differ by less than 1e-4 of the largest one),
/// all normals orthogonal to one direction (planes tangent to a cylinder), or no unique apex
[[nodiscard]] MRMESH_API std::optional<Cone3d> findBestConeByTangentPlanes( const std::vector<Vector3d>& points, const std::vector<Plane3d>& tangentPlanes );

/// torus with given axis fitted to the points: the circle of the tube section fitted by findBestCircle in the plane (distance to the axis, position along it),
/// so the minor radius is biased low for noisy points of a short arc of the tube, and collinear sections (e.g. of a cylinder) give its minimum-norm circle;
/// the result can have minorRadius > majorRadius (a spindle torus)
[[nodiscard]] MRMESH_API std::optional<Torus3d> findBestTorusWithAxis( const std::vector<Vector3d>& points, const Line3d& axis );

/// minimizes the sum of squared distances from the points to the torus by Levenberg-Marquardt method starting from given torus;
/// the result is never worse than given torus (with unit axis and non-negative radii), which is returned for fewer than 8 points;
/// it can be degenerate, so check its radii: majorRadius is about zero if a sphere fits the points better than any torus,
/// and both radii can be very large if the points are nearly on a cylinder, a cone or a plane
[[nodiscard]] MRMESH_API Torus3d refineTorus( const std::vector<Vector3d>& points, const Torus3d& torus );

} // namespace MR
