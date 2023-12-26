#include "MRPointToPlaneAligningTransform.h"
#include "MRVector3.h"
#include "MRAffineXf3.h"
#include "MRQuaternion.h"
#include "MRToFromEigen.h"
#include "MRGTest.h"
#include <Eigen/Eigenvalues>

namespace MR
{

// returns linear approximation of the rotation matrix, which is close to true rotation matrix for small angles
static Matrix3d approximateLinearRotationMatrixFromAngles( double alpha, double beta, double gamma )
{
    Vector3d row1 = {   1.0, -gamma,   beta };
    Vector3d row2 = { gamma,    1.0, -alpha };
    Vector3d row3 = { -beta,  alpha,    1.0 };

    return Matrix3d( row1, row2, row3 );
}

PointToPlaneAligningTransform::PointToPlaneAligningTransform( const AffineXf3d& aTransform )
    : approxTransform( aTransform )
{
    sum_A.setZero();
    sum_B.setZero();
}

void PointToPlaneAligningTransform::add(const Vector3d& s_, const Vector3d& d, const Vector3d& normal2, const double w)
{
    Vector3d s = approxTransform(s_);
    Vector3d n = normal2.normalized();
    double k_B = dot(d - s, n);
    double c[6];
    c[0] = n.z * s.y - n.y * s.z;
    c[1] = n.x * s.z - n.z * s.x;
    c[2] = n.y * s.x - n.x * s.y;
    c[3] = n.x;
    c[4] = n.y;
    c[5] = n.z;
    for (size_t i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < 6; j++)
        {
            sum_A.coeffRef(i, j) += w * c[i] * c[j];
        }

        sum_B(i, 0) += w * c[i] * k_B;
    }
}

void PointToPlaneAligningTransform::clear()
{
    sum_A.setZero();
    sum_B.setZero();
}

Vector3d PointToPlaneAligningTransform::findBestTranslation() const
{
    Eigen::Matrix<double, 3, 3> sum_A_restr;
    Eigen::Matrix<double, 3, 1> sum_B_restr;
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            sum_A_restr(i, j) = sum_A(i + 3, j + 3);
        }

        sum_B_restr(i, 0) = sum_B(i + 3, 0);
    }
    Eigen::LLT<Eigen::MatrixXd> chol(sum_A_restr);
    Eigen::VectorXd solution = chol.solve(sum_B_restr);
    return Vector3d{ solution.coeff(0), solution.coeff(1), solution.coeff(2) };
}

auto PointToPlaneAligningTransform::calculateAmendment() const -> Amendment
{
    Eigen::LLT<Eigen::MatrixXd> chol(sum_A);
    Eigen::VectorXd solution = chol.solve(sum_B);

    Amendment res;
    res.rotAngles = Vector3d{ solution.coeff( 0 ), solution.coeff( 1 ), solution.coeff( 2 ) };
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

    A(0,0) = k.transpose() * sum_A.topLeftCorner<3,3>() * k;

    const Eigen::Matrix<double, 3, 1> tk = sum_A.bottomLeftCorner<3,3>() * k;
    A.bottomLeftCorner<3,1>() = tk;
    A.topRightCorner<1,3>() = tk.transpose();

    A.bottomRightCorner<3,3>() = sum_A.bottomRightCorner<3,3>();

    b.topRows<1>() = k.transpose() * sum_B.topRows<3>();
    b.bottomRows<3>() = sum_B.bottomRows<3>();

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

    A.topLeftCorner<2,2>() = k.transpose() * sum_A.topLeftCorner<3,3>() * k;

    const Eigen::Matrix<double, 3, 2> tk = sum_A.bottomLeftCorner<3,3>() * k;
    A.bottomLeftCorner<3,2>() = tk;
    A.topRightCorner<2,3>() = tk.transpose();

    A.bottomRightCorner<3,3>() = sum_A.bottomRightCorner<3,3>();

    b.topRows<2>() = k.transpose() * sum_B.topRows<3>();
    b.bottomRows<3>() = sum_B.bottomRows<3>();

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

    return AffineXf3d(Quaterniond(ammendment.rotAngles, ammendment.rotAngles.length()), ammendment.shift) * approxTransform;
}

AffineXf3d PointToPlaneAligningTransform::findBestRigidXfFixedRotationAxis( const Vector3d & axis ) const
{
    const auto ammendment = calculateFixedAxisAmendment( axis );

    return AffineXf3d(Quaterniond(ammendment.rotAngles, ammendment.rotAngles.length()), ammendment.shift) * approxTransform;
}

AffineXf3d PointToPlaneAligningTransform::findBestRigidXfOrthogonalRotationAxis( const Vector3d & ort ) const
{
    const auto ammendment = calculateOrthogonalAxisAmendment( ort );

    return AffineXf3d(Quaterniond(ammendment.rotAngles, ammendment.rotAngles.length()), ammendment.shift) * approxTransform;
}

TEST( MRMesh, PointToPlaneIteration )
{
    std::vector<Vector3d> pInit, pTransformed, n, n2;
    pInit.resize( 10 );
    pTransformed.resize( 10 );
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

    double alpha = 0.15, beta = 0.23, gamma = -0.17;
    Matrix3d rotationMatrix = approximateLinearRotationMatrixFromAngles( alpha, beta, gamma );
    AffineXf3d xf1( rotationMatrix, Vector3d( 2., 3., -1. ) );
    for( int i = 0; i < 10; i++ )
    {
        pTransformed[i] = xf1( pInit[i] );
    }
    for( int i = 0; i < 3; i++ )
    {
        pTransformed[i] += n2[i];
    }

    PointToPlaneAligningTransform ptp1;
    for( int i = 0; i < 10; i++ )
    {
        ptp1.add( pInit[i], pTransformed[i], n[i] );
    }
    
    {
        const auto ammendment = ptp1.calculateAmendment();
        Matrix3d apprRotationMatrix = approximateLinearRotationMatrixFromAngles( ammendment.rotAngles.x, ammendment.rotAngles.y, ammendment.rotAngles.z );
        auto xf2 = AffineXf3d( apprRotationMatrix, ammendment.shift );
        EXPECT_NEAR( (xf1.A.x - xf2.A.x).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.y - xf2.A.y).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.z - xf2.A.z).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.b - xf2.b).length(), 0., 1e-13 );
    }

    {
        const auto ammendment = ptp1.calculateFixedAxisAmendment( 10.0 * Vector3d{ alpha, beta, gamma } );
        Matrix3d apprRotationMatrix = approximateLinearRotationMatrixFromAngles( ammendment.rotAngles.x, ammendment.rotAngles.y, ammendment.rotAngles.z );
        auto xf2 = AffineXf3d( apprRotationMatrix, ammendment.shift );
        EXPECT_NEAR( (xf1.A.x - xf2.A.x).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.y - xf2.A.y).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.z - xf2.A.z).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.b - xf2.b).length(), 0., 1e-13 );
    }

    {
        Vector3d axis = Vector3d{ alpha, beta, gamma };
        axis = cross( axis, axis.furthestBasisVector() );
        const auto ammendment = ptp1.calculateOrthogonalAxisAmendment( -12.0 * axis );
        Matrix3d apprRotationMatrix = approximateLinearRotationMatrixFromAngles( ammendment.rotAngles.x, ammendment.rotAngles.y, ammendment.rotAngles.z );
        auto xf2 = AffineXf3d( apprRotationMatrix, ammendment.shift );
        EXPECT_NEAR( (xf1.A.x - xf2.A.x).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.y - xf2.A.y).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.z - xf2.A.z).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.b - xf2.b).length(), 0., 1e-13 );
    }
}

} //namespace MR
