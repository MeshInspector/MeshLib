#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include <MRPch/MREigenCore.h>

namespace MR
{

/// \defgroup AligningTransformGroup Aligning Transform
/// \ingroup MathGroup
/// \{

/// This class and its main method can be used to solve the problem of 3D shape alignment.
/// This algorithm uses a point-to-plane error metric in which the object of minimization is the sum of
/// the squared distance between a point and the tangent plane at its correspondence point.
/// To use this technique it's need to have small rotation angles. So there is an approximate solution.
/// The result of this algorithm is the transformation of first points (p1) which aligns it to the second ones (p2).
class PointToPlaneAligningTransform
{
public:
    /// Constructor with the known approximation of the aligning transformation.
    PointToPlaneAligningTransform( const AffineXf3d& aTransform = {} ) : approxTransform_( aTransform ) {}

    /// Add a pair of corresponding points and the normal of the tangent plane at the second point.
    MRMESH_API void add(const Vector3d& p1, const Vector3d& p2, const Vector3d& normal2, const double w = 1. );

    /// Clear points and normals data
    void clear() { *this = {}; }

    /// Compute transformation as the solution to a least squares optimization problem:
    /// xf( p1_i ) = p2_i
    /// this version searches for best rigid body transformation
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidXf() const;

    /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidXfFixedRotationAxis( const Vector3d & axis ) const;

    /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidXfOrthogonalRotationAxis( const Vector3d& ort ) const;

    /// Returns only shift part relative to given approximation
    [[nodiscard]] MRMESH_API Vector3d findBestTranslation() const;

    struct Amendment
    {
        Vector3d rotAngles; ///< rotation angles relative to x,y,z axes
        Vector3d shift;
        double scale = 1;
    };

    /// Compute transformation relative to given approximation and return it as angles and shift
    [[nodiscard]] MRMESH_API Amendment calculateAmendment( bool scaleIsOne = true ) const;

    /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
    [[nodiscard]] MRMESH_API Amendment calculateFixedAxisAmendment( const Vector3d & axis ) const;

    /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
    [[nodiscard]] MRMESH_API Amendment calculateOrthogonalAxisAmendment( const Vector3d& ort ) const;

private:
    AffineXf3d approxTransform_;
    Eigen::Matrix<double, 7, 7> sumA_ = Eigen::Matrix<double, 7, 7>::Zero();
    Eigen::Vector<double, 7> sumB_ = Eigen::Vector<double, 7>::Zero();
};

/// \}

} //namespace MR
