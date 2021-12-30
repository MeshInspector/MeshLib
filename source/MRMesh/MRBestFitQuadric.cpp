#include "MRBestFitQuadric.h"
#include "MRVector4.h"
#include <Eigen/Dense>

namespace MR
{

QuadricApprox::QuadricApprox()
{
    accumA_.setZero();
    accumB_.setZero();
}

void QuadricApprox::addPoint( const Vector3d& point, double weight /*= 1.0 */ )
{
    Eigen::Matrix<double, 6, 1> vec;
    vec[0] = point.x * point.x;
    vec[1] = point.x * point.y;
    vec[2] = point.y * point.y;
    vec[3] = point.x;
    vec[4] = point.y;
    vec[5] = 1.0;

    Eigen::Matrix<double, 6, 1> wPt = weight * vec;

    accumA_ += ( wPt * vec.transpose() );
    accumB_ += ( wPt * point.z );
}

Eigen::Matrix<double, 6, 1> QuadricApprox::calcBestCoefficients() const
{
    return Eigen::Matrix<double, 6, 1>( accumA_.colPivHouseholderQr().solve( accumB_ ) );
}

Vector3d QuadricApprox::findZeroProjection( const Eigen::Matrix<double, 6, 1>& coefs )
{
    Eigen::Matrix2d A;
    A( 0, 0 ) = 2.0 * coefs[0] * coefs[5] + coefs[3] * coefs[3] + 1.0;
    A( 0, 1 ) = A( 1, 0 ) = coefs[1] * coefs[5] + coefs[3] * coefs[4];
    A( 1, 1 ) = 2.0 * coefs[2] * coefs[5] + coefs[4] * coefs[4] + 1.0;

    Eigen::Vector2d b;
    b[0] = -coefs[3] * coefs[5];
    b[1] = -coefs[4] * coefs[5];

    Eigen::Vector2d res = A.colPivHouseholderQr().solve( b );
    double z =
        coefs[0] * res[0] * res[0] +
        coefs[1] * res[0] * res[1] +
        coefs[2] * res[1] * res[1] +
        coefs[3] * res[0] +
        coefs[4] * res[1] +
        coefs[5];
    return Vector3d( res[0], res[1], z );
}

}