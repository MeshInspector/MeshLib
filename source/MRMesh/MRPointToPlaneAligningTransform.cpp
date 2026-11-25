#include "MRPointToPlaneAligningTransform.h"
#include "MRVector3.h"
#include "MRAffineXf3.h"
#include "MRQuaternion.h"
#include "MRSymMatrix4.h"
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

Eigen::Matrix<double, 7, 7> PointToPlaneAligningTransform::stabilizedA_() const
{
    assert( sumAIsSym_ );
    auto res = sumA_;
    if ( stabilizer_ > 0 )
    {
        const double rotStab = stabilizer_ * sumA_.topLeftCorner<3,3>().trace() / 3;
        const double shiftStab = stabilizer_ * sumA_.block<3,3>( 3, 3 ).trace() / 3;
        for ( int i = 0; i < 3; ++i )
        {
            res( i, i ) += rotStab;
            res( 3 + i, 3 + i ) += shiftStab;
        }
    }
    return res;
}

auto PointToPlaneAligningTransform::calculateAmendment() const -> RigidScaleXf3d
{
    const auto sumA = stabilizedA_();
    Eigen::LLT<Eigen::MatrixXd> chol( sumA.topLeftCorner<6,6>() );
    Eigen::VectorXd solution = chol.solve( sumB_.topRows<6>() - sumA.block<6,1>( 0, 6 ) );

    RigidScaleXf3d res;
    res.a = Vector3d{ solution.coeff( 0 ), solution.coeff( 1 ), solution.coeff( 2 ) };
    res.b = Vector3d{ solution.coeff( 3 ), solution.coeff( 4 ), solution.coeff( 5 ) };
    return res;
}

auto PointToPlaneAligningTransform::calculateAmendmentWithScale() const -> RigidScaleXf3d
{
    Eigen::LLT<Eigen::MatrixXd> chol( stabilizedA_() );
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

    const auto sumA = stabilizedA_();
    const auto k = toEigen( axis.normalized() );
    const Eigen::Matrix<double, 3, 1> tk = sumA.block<3,3>(3, 0) * k;

    SymMatrix4d A;
    A.xx = k.dot( sumA.topLeftCorner<3,3>() * k );
    A.xy = tk[0]; A.xz = tk[1]; A.xw = tk[2];
    A.yy = sumA( 3, 3 ); A.yz = sumA( 3, 4 ); A.yw = sumA( 3, 5 );
    A.zz = sumA( 4, 4 ); A.zw = sumA( 4, 5 );
    A.ww = sumA( 5, 5 );

    const Eigen::Matrix<double, 3, 1> bt = sumB_.middleRows<3>(3) - sumA.block<3,1>( 3, 6 );
    const Vector4d b{ k.dot( sumB_.topRows<3>() - sumA.block<3,1>( 0, 6 ) ), bt[0], bt[1], bt[2] };

    const Vector4d solution = A.solveLDLT( b );

    RigidScaleXf3d res;
    res.a = solution.x * fromEigen( k );
    res.b = Vector3d{ solution.y, solution.z, solution.w };
    return res;
}

auto PointToPlaneAligningTransform::calculateOrthogonalAxisAmendment( const Vector3d& ort ) const -> RigidScaleXf3d
{
    if ( ort.lengthSq() <= 0 )
        return calculateAmendment();

    const auto sumA = stabilizedA_();
    Eigen::Matrix<double, 5, 5> A;
    Eigen::Matrix<double, 5, 1> b;
    Eigen::Matrix<double, 3, 2> k;

    const auto [d0, d1] = ort.perpendicular();
    k.leftCols<1>() = toEigen( d0 );
    k.rightCols<1>() = toEigen( d1 );

    A.topLeftCorner<2,2>() = k.transpose() * sumA.topLeftCorner<3,3>() * k;

    const Eigen::Matrix<double, 3, 2> tk = sumA.block<3,3>(3, 0) * k;
    A.bottomLeftCorner<3,2>() = tk;
    A.topRightCorner<2,3>() = tk.transpose();

    A.bottomRightCorner<3,3>() = sumA.block<3,3>(3, 3);

    b.topRows<2>() = k.transpose() * ( sumB_.topRows<3>() - sumA.block<3,1>( 0, 6 ) );
    b.bottomRows<3>() = sumB_.middleRows<3>(3) - sumA.block<3,1>( 3, 6 );

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
    const auto sumA = stabilizedA_();
    Eigen::LLT<Eigen::MatrixXd> chol( sumA.block<3,3>(3, 3) );
    Eigen::VectorXd solution = chol.solve( sumB_.middleRows<3>( 3 )
        - ( sumA.block<3,3>( 3, 0 ) * toEigen( rotAngles ) + sumA.block<3,1>( 3, 6 ) ) * scale );
    return Vector3d{ solution.coeff(0), solution.coeff(1), solution.coeff(2) };
}

} //namespace MR
