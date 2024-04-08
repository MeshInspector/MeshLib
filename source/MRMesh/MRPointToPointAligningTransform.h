#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"

namespace MR
{

/// \defgroup AligningTransformGroup Aligning Transform
/// \ingroup MathGroup
/// \{

/// This class and its main method can be used to solve the problem well-known as the absolute orientation problem.
/// It means computing the transformation that aligns two sets of points for which correspondence is known.
class PointToPointAligningTransform
{
public:
    /// Add one pair of points in the set
    MRMESH_API void add( const Vector3d& p1, const Vector3d& p2, double w = 1 );

    /// Add one pair of points in the set
    void add( const Vector3f& p1, const Vector3f& p2, float w = 1 ) { add( Vector3d( p1 ), Vector3d( p2 ), w ); }

    /// Add another two sets of points.
    MRMESH_API void add( const PointToPointAligningTransform & other );

    /// Clear sets.
    void clear() { *this = {}; }

    /// returns weighted centroid of points p1 accumulated so far
    [[nodiscard]] Vector3d centroid1() const { return sum1_ / sumW_; }

    /// returns weighted centroid of points p2 accumulated so far
    [[nodiscard]] Vector3d centroid2() const { return sum2_ / sumW_; }

    /// returns summed weight of points accumulated so far
    [[nodiscard]] double totalWeight() const { return sumW_; }

    /// Compute transformation as the solution to a least squares formulation of the problem:
    /// xf( p1_i ) = p2_i
    /// this version searches for best rigid body transformation
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidXf() const;

    /// this version searches for best rigid body transformation with uniform scaling
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidScaleXf() const;

    /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidXfFixedRotationAxis( const Vector3d& axis ) const;

    /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
    [[nodiscard]] MRMESH_API AffineXf3d findBestRigidXfOrthogonalRotationAxis( const Vector3d& ort ) const;

    /// Simplified solution for translational part only
    [[nodiscard]] MRMESH_API Vector3d findBestTranslation() const;

private:
    struct BestRotation
    {
        Matrix3d rot;
        double err = 0; // larger value means more discrepancy between points after registration
    };
    /// finds rotation matrix that best aligns centered pairs of points
    BestRotation findPureRotation_() const;

private:
    Matrix3d sum12_ = Matrix3d::zero();
    Vector3d sum1_, sum2_;
    double sum11_ = 0; ///< used only for scale determination
    double sumW_ = 0;
};

/// \}

} // namespace MR
