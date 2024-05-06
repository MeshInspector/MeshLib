#pragma once

#include "MRMeshFwd.h"
#include "MRRigidScaleXf3.h"
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
    /// Add a pair of corresponding points and the normal of the tangent plane at the second point
    MRMESH_API void add( const Vector3d& p1, const Vector3d& p2, const Vector3d& normal2, double w = 1 );

    /// Add a pair of corresponding points and the normal of the tangent plane at the second point
    void add( const Vector3f& p1, const Vector3f& p2, const Vector3f& normal2, float w = 1 ) { add( Vector3d( p1 ), Vector3d( p2 ), Vector3d( normal2 ), w ); }

    /// this method must be called after add() and before constant find...()/calculate...() to make the matrix symmetric
    MRMESH_API void prepare();

    /// Clear points and normals data
    void clear() { *this = {}; }

    /// Compute transformation as the solution to a least squares optimization problem:
    /// xf( p1_i ) = p2_i
    /// this version searches for best rigid body transformation
    [[nodiscard]] AffineXf3d findBestRigidXf() const { return calculateAmendment().rigidScaleXf(); }

    /// this version searches for best rigid body transformation with uniform scaling
    [[nodiscard]] AffineXf3d findBestRigidScaleXf() const { return calculateAmendmentWithScale().rigidScaleXf(); }

    /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
    [[nodiscard]] AffineXf3d findBestRigidXfFixedRotationAxis( const Vector3d & axis ) const { return calculateFixedAxisAmendment( axis ).rigidScaleXf(); }

    /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
    [[nodiscard]] AffineXf3d findBestRigidXfOrthogonalRotationAxis( const Vector3d& ort ) const { return calculateOrthogonalAxisAmendment( ort ).rigidScaleXf(); }

    /// this version searches for best translational part of affine transformation with given linear part
    [[nodiscard]] MRMESH_API Vector3d findBestTranslation( Vector3d rotAngles = {}, double scale = 1 ) const;

    /// Compute transformation relative to given approximation and return it as angles and shift (scale = 1)
    [[nodiscard]] MRMESH_API RigidScaleXf3d calculateAmendment() const;
    
    /// Compute transformation relative to given approximation and return it as scale, angles and shift
    [[nodiscard]] MRMESH_API RigidScaleXf3d calculateAmendmentWithScale() const;

    /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
    [[nodiscard]] MRMESH_API RigidScaleXf3d calculateFixedAxisAmendment( const Vector3d & axis ) const;

    /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
    [[nodiscard]] MRMESH_API RigidScaleXf3d calculateOrthogonalAxisAmendment( const Vector3d& ort ) const;

private:
    Eigen::Matrix<double, 7, 7> sumA_ = Eigen::Matrix<double, 7, 7>::Zero();
    Eigen::Vector<double, 7> sumB_ = Eigen::Vector<double, 7>::Zero();
    bool sumAIsSym_ = true;
};

/// \}

} //namespace MR
