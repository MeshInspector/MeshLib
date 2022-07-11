#include "MRAligningTransform.h"
#include "MRVector3.h"
#include "MRAffineXf3.h"
#include "MRQuaternion.h"
#include "MRToFromEigen.h"
#include <Eigen/Eigenvalues>
#include "MRGTest.h"

namespace MR
{

static constexpr int X = 0;
static constexpr int Y = 1;
static constexpr int Z = 2;
static constexpr int C = 3;

PointToPointAligningTransform::PointToPointAligningTransform()
{
    summary.setZero();
}

inline Eigen::Matrix4d calculateMatrixP( const Eigen::Matrix4d & s )
{
    Eigen::Matrix4d p;
    p << s( X, X ) + s( Y, Y ) + s( Z, Z ),    s( Y, Z ) - s( Z, Y ),                s( Z, X ) - s( X, Z ),                s( X, Y ) - s( Y, X ),
         s( Y, Z ) - s( Z, Y ),                s( X, X ) - s( Y, Y ) - s( Z, Z ),    s( X, Y ) + s( Y, X ),                s( Z, X ) + s( X, Z ),
         s( Z, X ) - s( X, Z ),                s( X, Y ) + s( Y, X ),                s( Y, Y ) - s( X, X ) - s( Z, Z ),    s( Y, Z ) + s( Z, Y ),
         s( X, Y ) - s( Y, X ),                s( Z, X ) + s( X, Z ),                s( Y, Z ) + s( Z, Y ),                s( Z, Z ) - s( X, X ) - s( Y, Y );

    return p;
}

Eigen::Matrix3d caluclate2DimensionsP( const Eigen::Matrix4d& P, const Eigen::Vector4d& d1, const Eigen::Vector4d& d2 )
{
    Eigen::Matrix3d twoDimsP;
    const Eigen::Vector4d d0 = Eigen::Vector4d::Identity();
    const auto d0T = d0.transpose();
    const auto d1T = d1.transpose();
    const auto d2T = d2.transpose();
    twoDimsP <<
        d0T* P* d0, d0T* P* d1, d0T* P* d2,
        d1T* P* d0, d1T* P* d1, d1T* P* d2,
        d2T* P* d0, d2T* P* d1, d2T* P* d2;
    
    return twoDimsP;
}

void PointToPointAligningTransform::add( const Vector3d& p1, const Vector3d& p2, double w /*= 1.0*/ )
{
    auto VectorInColumn = []( const Vector3d& v )
    {
        return Eigen::Matrix<double, 4, 1>( v.x, v.y, v.z, 1);
    };

    summary += w * (VectorInColumn( p1 ) * VectorInColumn( p2 ).transpose());
}


void PointToPointAligningTransform::add( const PointToPointAligningTransform & other )
{
    summary += other.summary;
}


void PointToPointAligningTransform::clear()
{
    summary.setZero();
}

Vector3d PointToPointAligningTransform::centroid1() const
{
    Vector3d res{ summary( X, C ), summary( Y, C ), summary( Z, C ) };
    res /= totalWeight();
    return res;
}

Vector3d PointToPointAligningTransform::centroid2() const
{
    Vector3d res{ summary( C, X ), summary( C, Y ), summary( C, Z ) };
    res /= totalWeight();
    return res;
}

double PointToPointAligningTransform::totalWeight() const
{
    return summary( C, C );
}

AffineXf3d PointToPointAligningTransform::calculateTransformationMatrix() const
{
    // for more detail of this algorithm see paragraph "3.3 A solution involving unit quaternions" in 
    // http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    const auto centroid1 = this->centroid1();
    const auto centroid2 = this->centroid2();
    const auto totalWeight = this->totalWeight();

    Eigen::Matrix4d s = summary - totalWeight * toEigen( centroid1, 1.0 ) * toEigen( centroid2, 1.0 ).transpose();
    Eigen::Matrix4d p = calculateMatrixP( s );

    Eigen::Vector4d largestEigenVector = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>( p ).eigenvectors().col( 3 );
    Quaterniond q( largestEigenVector[0], largestEigenVector[1], largestEigenVector[2], largestEigenVector[3] );
    Matrix3d r{ q };
    const auto shift = centroid2 - r * centroid1;
    return AffineXf3d( r, shift );
}

AffineXf3d PointToPointAligningTransform::calculateFixedAxisRotation( const Vector3d & axis ) const
{
    if ( axis.lengthSq() <= 0 )
        return calculateTransformationMatrix();

    const auto centroid1 = this->centroid1();
    const auto centroid2 = this->centroid2();
    const auto totalWeight = this->totalWeight();

    const Eigen::Matrix4d s = summary - totalWeight * toEigen( centroid1, 1.0 ) * toEigen( centroid2, 1.0 ).transpose();

    const auto k = axis.normalized();

    // a = sum_i( dot( p2_i, cross( k, cross( k, p1_i ) ) )
    const auto a =
        ( k.x * k.x - 1 ) * s( X, X ) +
        ( k.y * k.y - 1 ) * s( Y, Y ) +
        ( k.z * k.z - 1 ) * s( Z, Z ) +
        ( k.x * k.y ) * ( s( X, Y ) + s( Y, X ) ) +
        ( k.x * k.z ) * ( s( X, Z ) + s( Z, X ) ) +
        ( k.y * k.z ) * ( s( Y, Z ) + s( Z, Y ) );

    // b = dot( k, sum_i cross( p1_i, p2_i ) )
    const auto b = 
        k.x * ( s( Y, Z ) - s( Z, Y ) ) +
        k.y * ( s( Z, X ) - s( X, Z ) ) +
        k.z * ( s( X, Y ) - s( Y, X ) );

    const auto phi = atan2( b, -a );

    const auto r = Matrix3d::rotation( k, phi );
    const auto shift = centroid2 - r * centroid1;
    return AffineXf3d( r, shift );
}

AffineXf3d PointToPointAligningTransform::calculateOrthogonalAxisRotation( const Vector3d& ort ) const
{
    // for more detail of this algorithm see paragraph "3.3 A solution involving unit quaternions" in 
    // http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    const auto centroid1 = this->centroid1();
    const auto centroid2 = this->centroid2();
    const auto totalWeight = this->totalWeight();

    Eigen::Matrix4d s = summary - totalWeight * toEigen( centroid1, 1.0 ) * toEigen( centroid2, 1.0 ).transpose();
    Eigen::Matrix4d p = calculateMatrixP( s );

    auto [d1, d2] = ort.perpendicular();

    Eigen::Matrix3d p2d = caluclate2DimensionsP( p, Eigen::Vector4d{0,d1[0],d1[1],d1[2]}, Eigen::Vector4d{0,d2[0],d2[1],d2[2]} );

    Eigen::Vector3d largestEigenVector = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>( p2d ).eigenvectors().col( 2 );
    Quaterniond q = Quaterniond{1,0,0,0}*largestEigenVector( 0 ) + Quaterniond{0,d1[0],d1[1],d1[2]}*largestEigenVector( 1 ) + Quaterniond{0,d2[0],d2[1],d2[2]}*largestEigenVector( 2 );

    Matrix3d r{q};
    const auto shift = centroid2 - r * centroid1;
    return AffineXf3d( r, shift );
}

Vector3d PointToPointAligningTransform::calculateTranslation() const
{
    return centroid2() - centroid1();
}


TEST( MRMesh, AligningTransform )
{
    PointToPointAligningTransform at1;

    Vector3d b( 1, 1, 1 );
    at1.add( Vector3d::plusX(), Vector3d::plusY() + b );
    at1.add( Vector3d::plusY(), Vector3d::plusZ() + b );
    at1.add( Vector3d::plusZ(), Vector3d::plusX() + b );
    auto xf1 = at1.calculateTransformationMatrix();

    ASSERT_NEAR( ( xf1( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );

    auto xf1_ = at1.calculateFixedAxisRotation( Vector3d{ 1, 1, 1 } );
    ASSERT_NEAR( ( xf1_( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1_( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1_( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );


    auto xf1__ = at1.calculateOrthogonalAxisRotation( Vector3d{1, 0, -1} );
    ASSERT_NEAR( ( xf1__( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1__( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1__( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );

    // Second test
    PointToPointAligningTransform at2;
    Vector3d p11, p12, p21, p22, p31, p32, p41, p42, p51, p52;
    p11 = { 2, 1, 1 }; p12 = { -7, -5, -1 };
    p21 = { 4, 1, 1 }; p22 = { -7, -7, -1 };
    p31 = { 4, 4, 1 }; p32 = { -7, -7, -4 };
    p41 = { 4, 4, 6 }; p42 = { -2, -7, -4 };
    p51 = { 3, 4, 6 }; p52 = { -2, -6, -4 };
    at2.add( p11, p12 );
    at2.add( p21, p22 );
    at2.add( p31, p32 );
    at2.add( p41, p42 );
    at2.add( p51, p52 );

    auto xf2 = at2.calculateTransformationMatrix();
    EXPECT_NEAR( ( xf2( p11 ) - p12 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p21 ) - p22 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p31 ) - p32 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p41 ) - p42 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p51 ) - p52 ).length(), 0., 1e-14 );

    auto xf2_ = at2.calculateFixedAxisRotation( Vector3d{ -1, 1, -1 } );
    EXPECT_NEAR( ( xf2_( p11 ) - p12 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2_( p21 ) - p22 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2_( p31 ) - p32 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2_( p41 ) - p42 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2_( p51 ) - p52 ).length(), 0., 1e-14 );

    // Third test
    PointToPointAligningTransform at3;
    p11 = { 10.0, 10.0, 20.0 }; p12 = { 16.0, 23.0, 10.0 };
    p21 = { 3.1, 4.3, 5.2 }; p22 = { 10.3, 8.2, 3.1 };
    p31 = { 0.1, 0.1, -0.1 }; p32 = { 6.1, 2.9, 0.1 };
    p41 = { -0.5, 4.2, 6.1 }; p42 = { 10.2, 9.1, -0.5 };
    p51 = { 3.8, 4.1, -1.5 }; p52 = { 10.1, 1.5, 3.8 };
    at3.add( p11, p12 );
    at3.add( p21, p22 );
    at3.add( p31, p32 );
    at3.add( p41, p42 );
    at3.add( p51, p52 );

    auto xf3 = at3.calculateTransformationMatrix();
    EXPECT_NEAR( ( xf3( p11 ) - p12 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p21 ) - p22 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p31 ) - p32 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p41 ) - p42 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p51 ) - p52 ).length(), 0., 1e-13 );

    auto xf3_ = at3.calculateFixedAxisRotation( Vector3d{ 1, 1, 1 } );
    EXPECT_NEAR( ( xf3_( p11 ) - p12 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p21 ) - p22 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p31 ) - p32 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p41 ) - p42 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p51 ) - p52 ).length(), 0., 1e-13 );
}

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

Vector3d PointToPlaneAligningTransform::calculateTranslation() const
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

AffineXf3d PointToPlaneAligningTransform::calculateSolution() const
{
    const auto ammendment = calculateAmendment();

    return AffineXf3d(Quaterniond(ammendment.rotAngles, ammendment.rotAngles.length()), ammendment.shift) * approxTransform;
}

AffineXf3d PointToPlaneAligningTransform::calculateFixedAxisRotation( const Vector3d & axis ) const
{
    const auto ammendment = calculateFixedAxisAmendment( axis );

    return AffineXf3d(Quaterniond(ammendment.rotAngles, ammendment.rotAngles.length()), ammendment.shift) * approxTransform;
}

AffineXf3d PointToPlaneAligningTransform::calculateOrthogonalAxisRotation( const Vector3d & ort ) const
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
