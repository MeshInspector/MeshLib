#pragma once
#include "MRMeshEigen.h"
#include "MRAffineXf3.h"

namespace MR
{

/// Accumulate points and make best quadric approximation
/// \details \f$ a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = z \f$
/// \ingroup MathGroup
class QuadricApprox
{
public:
    MRMESH_API QuadricApprox();

    /// Adds point to accumulation with weight
    MRMESH_API void addPoint( const Vector3d& point, double weight = 1.0 );

    /// Calculates best coefficients a, b, c, d
    /// a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = z
    MRMESH_API Eigen::Matrix<double, 6, 1> calcBestCoefficients() const;

    /// Finds projection of zero point to surface given by coefs:
    /// a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = z
    MRMESH_API static Vector3d findZeroProjection( const Eigen::Matrix<double, 6, 1>& coefs );
private:
    Eigen::Matrix<double, 6, 6> accumA_;
    Eigen::Matrix<double, 6, 1> accumB_;
};

}
