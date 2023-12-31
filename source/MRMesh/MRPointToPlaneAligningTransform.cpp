#include "MRPointToPlaneAligningTransform.h"
#include "MRVector3.h"
#include "MRAffineXf3.h"
#include "MRQuaternion.h"
#include "MRToFromEigen.h"
#include "MRGTest.h"
#include <Eigen/Cholesky> //LLT

namespace MR
{

void PointToPlaneAligningTransform::add( const Vector3d& s0, const Vector3d& d, const Vector3d& normal2, const double w )
{
    Vector3d s = approxTransform_( s0 );
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

    // copy values in lower-left part
    for (size_t i = 1; i < 7; i++)
        for (size_t j = 0; j < i; j++)
            sumA_(i, j) = sumA_(j, i);
}

void PointToPlaneAligningTransform::clear()
{
    sumA_ = Eigen::Matrix<double, 7, 7>::Zero();
    sumB_ = Eigen::Vector<double, 7>::Zero();
}

AffineXf3d PointToPlaneAligningTransform::Amendment::rigidScaleXf() const
{
    return { scale * Matrix3d( Quaterniond( rotAngles,  rotAngles.length() ) ), shift };
}

AffineXf3d PointToPlaneAligningTransform::Amendment::linearXf() const
{
    return { scale * Matrix3d::approximateLinearRotationMatrixFromEuler( rotAngles ), shift };
}

auto PointToPlaneAligningTransform::calculateAmendment() const -> Amendment
{
    Eigen::LLT<Eigen::MatrixXd> chol( sumA_.topLeftCorner<6,6>() );
    Eigen::VectorXd solution = chol.solve( sumB_.topRows<6>() - sumA_.block<6,1>( 0, 6 ) );

    Amendment res;
    res.rotAngles = Vector3d{ solution.coeff( 0 ), solution.coeff( 1 ), solution.coeff( 2 ) };
    res.shift =     Vector3d{ solution.coeff( 3 ), solution.coeff( 4 ), solution.coeff( 5 ) };
    return res;
}

auto PointToPlaneAligningTransform::calculateAmendmentWithScale() const -> Amendment
{
    Eigen::LLT<Eigen::MatrixXd> chol( sumA_ );
    Eigen::VectorXd solution = chol.solve( sumB_ );

    Amendment res;
    res.scale = solution.coeff( 6 );
    res.rotAngles = Vector3d{ solution.coeff( 0 ), solution.coeff( 1 ), solution.coeff( 2 ) } / res.scale;
    res.shift =     Vector3d{ solution.coeff( 3 ), solution.coeff( 4 ), solution.coeff( 5 ) };
    return res;
}

auto PointToPlaneAligningTransform::calculateFixedAxisAmendment( const Vector3d & axis ) const -> Amendment
{
    if ( axis.lengthSq() <= 0 )
        return calculateAmendment();

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

    Amendment res;
    res.rotAngles = solution.coeff( 0 ) * fromEigen( k );
    res.shift =     Vector3d{ solution.coeff( 1 ), solution.coeff( 2 ), solution.coeff( 3 ) };
    return res;
}

auto PointToPlaneAligningTransform::calculateOrthogonalAxisAmendment( const Vector3d& ort ) const -> Amendment
{
    if ( ort.lengthSq() <= 0 )
        return calculateAmendment();

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

    Amendment res;
    res.rotAngles = solution.coeff( 0 ) * fromEigen( Eigen::Vector3d{ k.leftCols<1>() } ) 
                  + solution.coeff( 1 ) * fromEigen( Eigen::Vector3d{ k.rightCols<1>() } );
    res.shift =     Vector3d{ solution.coeff( 2 ), solution.coeff( 3 ), solution.coeff( 4 ) };
    return res;
}

AffineXf3d PointToPlaneAligningTransform::findBestRigidXf() const
{
    const auto ammendment = calculateAmendment();
    return ammendment.rigidScaleXf() * approxTransform_;
}

AffineXf3d PointToPlaneAligningTransform::findBestRigidScaleXf() const
{
    const auto ammendment = calculateAmendmentWithScale();
    return ammendment.rigidScaleXf() * approxTransform_;
}

AffineXf3d PointToPlaneAligningTransform::findBestRigidXfFixedRotationAxis( const Vector3d & axis ) const
{
    const auto ammendment = calculateFixedAxisAmendment( axis );
    return ammendment.rigidScaleXf() * approxTransform_;
}

AffineXf3d PointToPlaneAligningTransform::findBestRigidXfOrthogonalRotationAxis( const Vector3d & ort ) const
{
    const auto ammendment = calculateOrthogonalAxisAmendment( ort );
    return ammendment.rigidScaleXf() * approxTransform_;
}

Vector3d PointToPlaneAligningTransform::findBestTranslation() const
{
    Eigen::LLT<Eigen::MatrixXd> chol( sumA_.block<3,3>(3, 3) );
    Eigen::VectorXd solution = chol.solve( sumB_.middleRows<3>( 3 ) - sumA_.block<3,1>( 3, 6 ) );
    return Vector3d{ solution.coeff(0), solution.coeff(1), solution.coeff(2) };
}

TEST( MRMesh, PointToPlaneIteration )
{
    std::vector<Vector3d> pInit, n, n2;
    pInit.resize( 10 );
    n.resize( 10 );
    n2.resize( 3 );

    pInit[0]  = {   1.0,   1.0, -5.0 }; n[0] = {  0.0,  0.0, -1.0 }; n2[0] = { 0.1, -0.1,  0.0 };
    pInit[1]  = {  14.0,   1.0,  1.0 }; n[1] = {  1.0,  0.1,  1.0 }; n2[1] = { 0.3,  0.0, -0.3 };
    pInit[2]  = {   1.0,  14.0,  2.0 }; n[2] = {  0.1,  1.0,  1.2 }; n2[2] = { 0.0, -0.6,  0.5 };
    pInit[3]  = { -11.0,   2.0,  3.0 }; n[3] = { -1.0,  0.1,  1.0 };
    pInit[4]  = {   1.0, -11.0,  4.0 }; n[4] = {  0.1, -1.1,  1.1 };
    pInit[5]  = {   1.0,   2.0,  8.0 }; n[5] = {  0.1,  0.1,  1.0 };
    pInit[6]  = {   2.0,   1.0, -5.0 }; n[6] = {  0.1,  0.0, -1.0 };
    pInit[7]  = {  15.0,   1.5,  1.0 }; n[7] = {  1.1,  0.1,  1.0 };
    pInit[8]  = {   1.5,  15.0,  2.0 }; n[8] = {  0.1,  1.0,  1.2 };
    pInit[9]  = { -11.0,   2.5,  3.1 }; n[9] = { -1.1,  0.1,  1.1 };

    auto preparePt2Pl = [&]( const AffineXf3d & xf )
    {
        std::vector<Vector3d> pTransformed( 10 );

        for( int i = 0; i < 10; i++ )
            pTransformed[i] = xf( pInit[i] );
        for( int i = 0; i < 3; i++ )
            pTransformed[i] += n2[i];

        PointToPlaneAligningTransform ptp;
        for( int i = 0; i < 10; i++ )
            ptp.add( pInit[i], pTransformed[i], n[i] );
        return ptp;
    };

    double alpha = 0.15, beta = 0.23, gamma = -0.17;
    const Vector3d eulerAngles{ alpha, beta, gamma };
    const auto [e1, e2] = eulerAngles.perpendicular();
    Matrix3d rotationMatrix = Matrix3d::approximateLinearRotationMatrixFromEuler( eulerAngles );
    const Vector3d b( 2., 3., -1. );
    AffineXf3d xf1( rotationMatrix, b );

    const auto ptp1 = preparePt2Pl( xf1 );
    constexpr double eps = 3e-13;

    {
        const auto ammendment = ptp1.calculateAmendment();
        EXPECT_EQ( ammendment.scale, 1 );
        auto xf2 = ammendment.linearXf();
        EXPECT_NEAR( ( xf1.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf2.b ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateAmendmentWithScale();
        EXPECT_NEAR( ammendment.scale, 1., 1e-13 );
        auto xf2 = ammendment.linearXf();
        EXPECT_NEAR( ( xf1.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf2.b ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateFixedAxisAmendment( 10.0 * eulerAngles );
        EXPECT_EQ( ammendment.scale, 1 );
        EXPECT_NEAR( cross( ammendment.rotAngles, eulerAngles.normalized() ).length(), 0., eps );
        auto xf2 = ammendment.linearXf();
        EXPECT_NEAR( ( xf1.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf2.b ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateFixedAxisAmendment( e1 );
        EXPECT_EQ( ammendment.scale, 1 );
        EXPECT_NEAR( cross( ammendment.rotAngles, e1 ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateOrthogonalAxisAmendment( -12.0 * e1 );
        EXPECT_EQ( ammendment.scale, 1 );
        EXPECT_NEAR( dot( ammendment.rotAngles, e1 ), 0., eps );
        auto xf2 = ammendment.linearXf();
        EXPECT_NEAR( ( xf1.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf2.b ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateOrthogonalAxisAmendment( 12.0 * e2 );
        EXPECT_EQ( ammendment.scale, 1 );
        EXPECT_NEAR( dot( ammendment.rotAngles, e2 ), 0., eps );
        auto xf2 = ammendment.linearXf();
        EXPECT_NEAR( ( xf1.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf1.b - xf2.b ).length(), 0., eps );
    }

    {
        const auto ammendment = ptp1.calculateOrthogonalAxisAmendment( eulerAngles );
        EXPECT_EQ( ammendment.scale, 1 );
        EXPECT_NEAR( dot( ammendment.rotAngles, eulerAngles.normalized() ), 0., eps );
    }

    {
        const double scale = 0.5;
        AffineXf3d xf2( scale * rotationMatrix, b );
        const auto ptp2 = preparePt2Pl( xf2 );
        const auto ammendment = ptp2.calculateAmendmentWithScale();
        EXPECT_NEAR( ammendment.scale, scale, 1e-13 );
        auto xf3 = ammendment.linearXf();
        EXPECT_NEAR( ( xf3.A - xf2.A ).norm(), 0., eps );
        EXPECT_NEAR( ( xf3.b - xf2.b ).length(), 0., eps );
    }

    {
        AffineXf3d xf( {}, b );
        const auto ptp = preparePt2Pl( xf );
        const auto shift = ptp.findBestTranslation();
        EXPECT_NEAR( ( b - shift ).length(), 0., eps );
    }
}

} //namespace MR
