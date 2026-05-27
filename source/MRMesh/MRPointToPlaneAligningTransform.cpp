#include "MRPointToPlaneAligningTransform.h"
#include "MRVector3.h"
#include "MRAffineXf3.h"
#include "MRQuaternion.h"
#include "MRToFromEigen.h"
#include <Eigen/Cholesky> //LLT

namespace MR
{

void PointToPlaneAligningTransform::add( const Vector3d& s, const Vector3d& d, const Vector3d& normal2, const double w )
{
    Vector3d n = normal2.normalized();
    double k_B = dot( d, n );
    double c[7];
    // https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
    c[0] = n.z * s.y - n.y * s.z;
    c[1] = n.x * s.z - n.z * s.x;
    c[2] = n.y * s.x - n.x * s.y;
    c[3] = n.x;
    c[4] = n.y;
    c[5] = n.z;
    c[6] = dot( s, n );
    // update upper-right part of sumA_
    for (size_t i = 0; i < 7; i++)
    {
        for (size_t j = i; j < 7; j++)
            sumA_(i, j) += w * c[i] * c[j];

        sumB_(i) += w * c[i] * k_B;
    }
    sumAIsSym_ = false;
}

void PointToPlaneAligningTransform::prepare()
{
    if ( sumAIsSym_ )
        return;
    // copy values in lower-left part
    for (size_t i = 1; i < 7; i++)
        for (size_t j = 0; j < i; j++)
            sumA_(i, j) = sumA_(j, i);
    sumAIsSym_ = true;
}

auto PointToPlaneAligningTransform::calculateAmendment() const -> RigidScaleXf3d
{
    assert( sumAIsSym_ );
    Eigen::LLT<Eigen::MatrixXd> chol( sumA_.topLeftCorner<6,6>() );
    Eigen::VectorXd solution = chol.solve( sumB_.topRows<6>() - sumA_.block<6,1>( 0, 6 ) );

    RigidScaleXf3d res;
    res.a = Vector3d{ solution.coeff( 0 ), solution.coeff( 1 ), solution.coeff( 2 ) };
    res.b = Vector3d{ solution.coeff( 3 ), solution.coeff( 4 ), solution.coeff( 5 ) };
    return res;
}

auto PointToPlaneAligningTransform::calculateAmendmentWithScale() const -> RigidScaleXf3d
{
    assert( sumAIsSym_ );
    Eigen::LLT<Eigen::MatrixXd> chol( sumA_ );
    Eigen::VectorXd solution = chol.solve( sumB_ );

    RigidScaleXf3d res;
    res.s = solution.coeff( 6 );
    res.a = Vector3d{ solution.coeff( 0 ), solution.coeff( 1 ), solution.coeff( 2 ) } / res.s;
    res.b = Vector3d{ solution.coeff( 3 ), solution.coeff( 4 ), solution.coeff( 5 ) };
    return res;
}

auto PointToPlaneAligningTransform::calculateFixedAxisAmendment( const Vector3d & axis ) const -> RigidScaleXf3d
{
    if ( axis.lengthSq() <= 0 )
        return calculateAmendment();

    assert( sumAIsSym_ );
    Eigen::Matrix<double, 4, 4> A;
    Eigen::Matrix<double, 4, 1> b;

    const auto k = toEigen( axis.normalized() );

    A(0,0) = k.transpose() * sumA_.topLeftCorner<3,3>() * k;

    const Eigen::Matrix<double, 3, 1> tk = sumA_.block<3,3>(3, 0) * k;
    A.bottomLeftCorner<3,1>() = tk;
    A.topRightCorner<1,3>() = tk.transpose();

    A.bottomRightCorner<3,3>() = sumA_.block<3,3>(3, 3);

    b.topRows<1>() = k.transpose() * ( sumB_.topRows<3>() - sumA_.block<3,1>( 0, 6 ) );
    b.bottomRows<3>() = sumB_.middleRows<3>(3) - sumA_.block<3,1>( 3, 6 );

    Eigen::LLT<Eigen::MatrixXd> chol(A);
    Eigen::VectorXd solution = chol.solve(b);

    RigidScaleXf3d res;
    res.a = solution.coeff( 0 ) * fromEigen( k );
    res.b = Vector3d{ solution.coeff( 1 ), solution.coeff( 2 ), solution.coeff( 3 ) };
    return res;
}

auto PointToPlaneAligningTransform::calculateOrthogonalAxisAmendment( const Vector3d& ort ) const -> RigidScaleXf3d
{
    if ( ort.lengthSq() <= 0 )
        return calculateAmendment();

    assert( sumAIsSym_ );
    Eigen::Matrix<double, 5, 5> A;
    Eigen::Matrix<double, 5, 1> b;
    Eigen::Matrix<double, 3, 2> k;

    const auto [d0, d1] = ort.perpendicular();
    k.leftCols<1>() = toEigen( d0 );
    k.rightCols<1>() = toEigen( d1 );

    A.topLeftCorner<2,2>() = k.transpose() * sumA_.topLeftCorner<3,3>() * k;

    const Eigen::Matrix<double, 3, 2> tk = sumA_.block<3,3>(3, 0) * k;
    A.bottomLeftCorner<3,2>() = tk;
    A.topRightCorner<2,3>() = tk.transpose();

    A.bottomRightCorner<3,3>() = sumA_.block<3,3>(3, 3);

    b.topRows<2>() = k.transpose() * ( sumB_.topRows<3>() - sumA_.block<3,1>( 0, 6 ) );
    b.bottomRows<3>() = sumB_.middleRows<3>(3) - sumA_.block<3,1>( 3, 6 );

    Eigen::LLT<Eigen::MatrixXd> chol(A);
    Eigen::VectorXd solution = chol.solve(b);

    RigidScaleXf3d res;
    res.a = solution.coeff( 0 ) * fromEigen( Eigen::Vector3d{ k.leftCols<1>() } ) 
                  + solution.coeff( 1 ) * fromEigen( Eigen::Vector3d{ k.rightCols<1>() } );
    res.b = Vector3d{ solution.coeff( 2 ), solution.coeff( 3 ), solution.coeff( 4 ) };
    return res;
}

Vector3d PointToPlaneAligningTransform::findBestTranslation( Vector3d rotAngles, double scale ) const
{
    assert( sumAIsSym_ );
    Eigen::LLT<Eigen::MatrixXd> chol( sumA_.block<3,3>(3, 3) );
    Eigen::VectorXd solution = chol.solve( sumB_.middleRows<3>( 3 )
        - ( sumA_.block<3,3>( 3, 0 ) * toEigen( rotAngles ) + sumA_.block<3,1>( 3, 6 ) ) * scale );
    return Vector3d{ solution.coeff(0), solution.coeff(1), solution.coeff(2) };
}

} //namespace MR
