#pragma once
#include "MRMeshFwd.h"
#include "MRAffineXf3.h"

#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#if __GNUC__ == 12
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
#endif

#include <Eigen/Core>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#pragma warning(pop)

namespace MR
{

/// \defgroup AligningTransformGroup
/// \ingroup MathGroup
/// \{

/// This class and its main method can be used to solve the problem well-known as the absolute orientation problem.
/// It means computing the transformation that aligns two sets of points for which correspondence is known.
class PointToPointAligningTransform
{
public:
    /// Default constructor
    MRMESH_API PointToPointAligningTransform();

    /// Add one pair of points in the set.
    MRMESH_API void add( const Vector3d& p1, const Vector3d& p2, double w = 1.0 );
    /// Add another two sets of points.
    MRMESH_API void add( const PointToPointAligningTransform & other );
    /// Clear sets.
    MRMESH_API void clear();

    /// returns weighted centroid of points p1 accumulated so far
    MRMESH_API Vector3d centroid1() const;
    /// returns weighted centroid of points p2 accumulated so far
    MRMESH_API Vector3d centroid2() const;
    /// returns summed weight of points accumulated so far
    MRMESH_API double totalWeight() const;

    /// Compute transformation as the solution to a least squares formulation of the problem:
    /// xf( p1_i ) = p2_i
    MRMESH_API AffineXf3d calculateTransformationMatrix() const;
    /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
    MRMESH_API AffineXf3d calculateFixedAxisRotation( const Vector3d& axis ) const;
    /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
    MRMESH_API AffineXf3d calculateOrthogonalAxisRotation( const Vector3d& ort ) const;
    /// Simplified solution for translational part only
    MRMESH_API Vector3d calculateTranslation() const;

private:
    Eigen::Matrix<double, 4, 4> summary;
};


/// This class and its main method can be used to solve the problem of 3D shape alignment.
/// This algorithm uses a point-to-plane error metric in which the object of minimization is the sum of
/// the squared distance between a point and the tangent plane at its correspondence point.
/// To use this technique it's need to have small rotation angles. So there is an approximate solution.
/// The result of this algorithm is the transformation of first points (p1) which aligns it to the second ones (p2).
class PointToPlaneAligningTransform
{
public:
    /// Constructor with the known approximation of the aligning transformation.
    MRMESH_API PointToPlaneAligningTransform( const AffineXf3d& aTransform = {} );

    /// Add a pair of corresponding points and the normal of the tangent plane at the second point.
    MRMESH_API void add(const Vector3d& p1, const Vector3d& p2, const Vector3d& normal2, const double w = 1.);
    /// Clear points and normals data
    MRMESH_API void clear();

    /// Compute transformation as the solution to a least squares optimization problem:
    /// xf( p1_i ) = p2_i
    MRMESH_API AffineXf3d calculateSolution() const;
    /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
    MRMESH_API AffineXf3d calculateFixedAxisRotation( const Vector3d & axis ) const;
    /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
    MRMESH_API AffineXf3d calculateOrthogonalAxisRotation( const Vector3d& ort ) const;
    /// Returns only shift part relative to given approximation
    MRMESH_API Vector3d calculateTranslation() const;

    struct Amendment
    {
        Vector3d rotAngles; ///< rotation angles relative to x,y,z axes
        Vector3d shift;
    };

    /// Compute transformation relative to given approximation and return it as angles and shift
    MRMESH_API Amendment calculateAmendment() const;
    /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
    MRMESH_API Amendment calculateFixedAxisAmendment( const Vector3d & axis ) const;
    /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
    MRMESH_API Amendment calculateOrthogonalAxisAmendment( const Vector3d& ort ) const;

private:
    AffineXf3d approxTransform;
    Eigen::Matrix<double, 6, 6> sum_A;
    Eigen::Matrix<double, 6, 1> sum_B;
};

/// \}

}
