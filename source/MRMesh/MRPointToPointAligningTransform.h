#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include "MRMatrix4.h"

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
    /// Default constructor
    MRMESH_API PointToPointAligningTransform();

    /// Add one pair of points in the set.
    [[nodiscard]] MRMESH_API void add( const Vector3d& p1, const Vector3d& p2, double w = 1.0 );
    /// Add another two sets of points.
    [[nodiscard]] MRMESH_API void add( const PointToPointAligningTransform & other );
    /// Clear sets.
    [[nodiscard]] MRMESH_API void clear();

    /// returns weighted centroid of points p1 accumulated so far
    [[nodiscard]] MRMESH_API Vector3d centroid1() const;
    /// returns weighted centroid of points p2 accumulated so far
    [[nodiscard]] MRMESH_API Vector3d centroid2() const;
    /// returns summed weight of points accumulated so far
    [[nodiscard]] double totalWeight() const { return summary_.w.w; }

    /// Compute transformation as the solution to a least squares formulation of the problem:
    /// xf( p1_i ) = p2_i
    [[nodiscard]] MRMESH_API AffineXf3d calculateTransformationMatrix() const;
    /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
    [[nodiscard]] MRMESH_API AffineXf3d calculateFixedAxisRotation( const Vector3d& axis ) const;
    /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
    [[nodiscard]] MRMESH_API AffineXf3d calculateOrthogonalAxisRotation( const Vector3d& ort ) const;
    /// Simplified solution for translational part only
    [[nodiscard]] MRMESH_API Vector3d calculateTranslation() const;

private:
    Matrix4d summary_;
};

/// \}

} // namespace MR
