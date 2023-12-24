#include "MRPointToPointAligningTransform.h"
#include "MRVector3.h"
#include "MRAffineXf3.h"
#include "MRQuaternion.h"
#include "MRToFromEigen.h"
#include "MRGTest.h"
#include <Eigen/Eigenvalues>

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

} //namespace MR
