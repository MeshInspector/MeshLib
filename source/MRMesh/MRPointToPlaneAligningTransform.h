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
    /// Constructor with the known approximation of the aligning transformation
    explicit PointToPlaneAligningTransform( const AffineXf3d& aTransform = {} ) : approxTransform_( aTransform ) {}

    /// Add a pair of corresponding points and the normal of the tangent plane at the second point
    MRMESH_API void add( const Vector3d& p1, const Vector3d& p2, const Vector3d& normal2, double w = 1 );

    /// Add a pair of corresponding points and the normal of the tangent plane at the second point
    void add( const Vector3f& p1, const Vector3f& p2, const Vector3f& normal2, float w = 1 ) { add( Vector3d( p1 ), Vector3d( p2 ), Vector3d( normal2 ), w ); }

    /// Clear points and normals data
    MRMESH_API void clear();

    /// Compute transformation as the solution to a least squares optimization problem:
    /// xf( p1_i ) = p2_i
    /// this version searches for best rigid body transformation
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidXf() const;

    /// this version searches for best rigid body transformation with uniform scaling
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidScaleXf() const;

    /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidXfFixedRotationAxis( const Vector3d & axis ) const;

    /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidXfOrthogonalRotationAxis( const Vector3d& ort ) const;

    /// this version searches for best translational part of affine transformation with given linear part
    [[nodiscard]] MRMESH_API Vector3d findBestTranslation( Vector3d rotAngles = {}, double scale = 1 ) const;

    struct Amendment
    {
        Vector3d rotAngles; ///< rotation angles relative to x,y,z axes
        Vector3d shift;
        double scale = 1;

        /// converts this amendment into rigid (with scale) transformation, which non-linearly depends on angles
        [[nodiscard]] MRMESH_API AffineXf3d rigidScaleXf() const;

        /// converts this amendment into not-rigid transformation but with matrix, which linearly depends on angles
        [[nodiscard]] MRMESH_API AffineXf3d linearXf() const;
    };

    /// Compute transformation relative to given approximation and return it as angles and shift (scale = 1)
    [[nodiscard]] MRMESH_API Amendment calculateAmendment() const;

    /// Compute transformation relative to given approximation and return it as scale, angles and shift
    [[nodiscard]] MRMESH_API Amendment calculateAmendmentWithScale() const;

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
