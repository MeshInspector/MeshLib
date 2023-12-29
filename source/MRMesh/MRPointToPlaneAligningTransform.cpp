#include "MRPointToPlaneAligningTransform.h"
#include "MRVector3.h"
#include "MRAffineXf3.h"
#include "MRQuaternion.h"
#include "MRToFromEigen.h"
#include "MRGTest.h"
#include <Eigen/Eigenvalues>

namespace MR
{

void PointToPlaneAligningTransform::add(const Vector3d& s0, const Vector3d& d, const Vector3d& normal2, const double w)
{
    Vector3d s = approxTransform_(s0);
    Vector3d n = normal2.normalized();
    double k_B = dot(d, n); // dot(d - s, n)
    double c[7];
    // https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
    c[0] = n.z * s.y - n.y * s.z;
    c[1] = n.x * s.z - n.z * s.x;
    c[2] = n.y * s.x - n.x * s.y;
    c[3] = n.x;
    c[4] = n.y;
    c[5] = n.z;
    c[6] = dot( s, n );
    for (size_t i = 0; i < 7; i++)
    {
        for (size_t j = 0; j < 7; j++)
        {
            sumA_(i, j) += w * c[i] * c[j];
        }

        sumB_(i) += w * c[i] * k_B;
    }
}

Vector3d PointToPlaneAligningTransform::findBestTranslation() const
{
    Eigen::Matrix<double, 3, 3> sum_A_restr;
    Eigen::Matrix<double, 3, 1> sum_B_restr;
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            sum_A_restr(i, j) = sumA_(i + 3, j + 3);
        }

        sum_B_restr(i) = sumB_(i + 3);
    }
    Eigen::LLT<Eigen::MatrixXd> chol(sum_A_restr);
    Eigen::VectorXd solution = chol.solve(sum_B_restr);
    return Vector3d{ solution.coeff(0), solution.coeff(1), solution.coeff(2) };
}

auto PointToPlaneAligningTransform::calculateAmendment( bool scaleIsOne ) const -> Amendment
{
    Amendment res;
    if ( scaleIsOne )
    {
        Eigen::LLT<Eigen::MatrixXd> chol( sumA_.topLeftCorner<6,6>() );
        Eigen::VectorXd solution = chol.solve( sumB_.topRows<6>() );
        
        res.rotAngles = Vector3d{ solution.coeff( 0 ), solution.coeff( 1 ), solution.coeff( 2 ) };
        res.shift =     Vector3d{ solution.coeff( 3 ), solution.coeff( 4 ), solution.coeff( 5 ) };
    }
    else
    {
        Eigen::LLT<Eigen::MatrixXd> chol( sumA_ );
        Eigen::VectorXd solution = chol.solve( sumB_ );
        
        res.scale = solution.coeff( 6 );
        res.rotAngles = Vector3d{ solution.coeff( 0 ), solution.coeff( 1 ), solution.coeff( 2 ) } / res.scale;
        res.shift =     Vector3d{ solution.coeff( 3 ), solution.coeff( 4 ), solution.coeff( 5 ) };
    }
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

    b.topRows<1>() = k.transpose() * sumB_.topRows<3>();
    b.bottomRows<3>() = sumB_.middleRows<3>(3);

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

    b.topRows<2>() = k.transpose() * sumB_.topRows<3>();
    b.bottomRows<3>() = sumB_.middleRows<3>(3);

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

    return AffineXf3d(Quaterniond(ammendment.rotAngles, ammendment.rotAngles.length()), ammendment.shift) * approxTransform_;
}

AffineXf3d PointToPlaneAligningTransform::findBestRigidXfFixedRotationAxis( const Vector3d & axis ) const
{
    const auto ammendment = calculateFixedAxisAmendment( axis );

    return AffineXf3d(Quaterniond(ammendment.rotAngles, ammendment.rotAngles.length()), ammendment.shift) * approxTransform_;
}

AffineXf3d PointToPlaneAligningTransform::findBestRigidXfOrthogonalRotationAxis( const Vector3d & ort ) const
{
    const auto ammendment = calculateOrthogonalAxisAmendment( ort );

    return AffineXf3d(Quaterniond(ammendment.rotAngles, ammendment.rotAngles.length()), ammendment.shift) * approxTransform_;
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
    Matrix3d rotationMatrix = Matrix3d::approximateLinearRotationMatrixFromEuler( { alpha, beta, gamma } );
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
    
/*    {
        const auto ammendment = ptp1.calculateAmendment();
        EXPECT_EQ( ammendment.scale, 1 );
        Matrix3d apprRotationMatrix = Matrix3d::approximateLinearRotationMatrixFromEuler( ammendment.rotAngles );
        auto xf2 = AffineXf3d( apprRotationMatrix, ammendment.shift );
        EXPECT_NEAR( (xf1.A.x - xf2.A.x).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.y - xf2.A.y).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.z - xf2.A.z).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.b - xf2.b).length(), 0., 1e-13 );
    }*/

    {
        const auto ammendment = ptp1.calculateAmendment( false );
        EXPECT_NEAR( ammendment.scale, 1., 1e-13 );
        Matrix3d apprRotationMatrix = Matrix3d::approximateLinearRotationMatrixFromEuler( ammendment.rotAngles );
        auto xf2 = AffineXf3d( apprRotationMatrix, ammendment.shift );
        EXPECT_NEAR( (xf1.A.x - xf2.A.x).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.y - xf2.A.y).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.z - xf2.A.z).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.b - xf2.b).length(), 0., 1e-13 );
    }

    {
        const double scale = 0.5;
        AffineXf3d xf2( scale * rotationMatrix, Vector3d( 2., 3., -1. ) );
        for( int i = 0; i < 10; i++ )
        {
            pTransformed[i] = xf2( pInit[i] );
        }
        for( int i = 0; i < 3; i++ )
        {
            pTransformed[i] += n2[i];
        }
        PointToPlaneAligningTransform ptp2;
        for( int i = 0; i < 10; i++ )
        {
            ptp2.add( pInit[i], pTransformed[i], n[i] );
        }
        const auto ammendment = ptp2.calculateAmendment( false );
        EXPECT_NEAR( ammendment.scale, scale, 1e-13 );
        Matrix3d apprRotationMatrix = ammendment.scale * Matrix3d::approximateLinearRotationMatrixFromEuler( ammendment.rotAngles );
        auto xf3 = AffineXf3d( apprRotationMatrix, ammendment.shift );
        EXPECT_NEAR( (xf3.A.x - xf2.A.x).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf3.A.y - xf2.A.y).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf3.A.z - xf2.A.z).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf3.b - xf2.b).length(), 0., 1e-13 );
    }

/*    {
        const auto ammendment = ptp1.calculateFixedAxisAmendment( 10.0 * Vector3d{ alpha, beta, gamma } );
        EXPECT_EQ( ammendment.scale, 1 );
        Matrix3d apprRotationMatrix = Matrix3d::approximateLinearRotationMatrixFromEuler( ammendment.rotAngles );
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
        EXPECT_EQ( ammendment.scale, 1 );
        Matrix3d apprRotationMatrix = Matrix3d::approximateLinearRotationMatrixFromEuler( ammendment.rotAngles );
        auto xf2 = AffineXf3d( apprRotationMatrix, ammendment.shift );
        EXPECT_NEAR( (xf1.A.x - xf2.A.x).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.y - xf2.A.y).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.A.z - xf2.A.z).length(), 0., 1e-13 );
        EXPECT_NEAR( (xf1.b - xf2.b).length(), 0., 1e-13 );
    }*/
}

} //namespace MR
