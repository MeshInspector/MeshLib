#include "MRPointToPointAligningTransform.h"
#include "MRVector3.h"
#include "MRSymMatrix3.h"
#include "MRMatrix4.h"
#include "MRQuaternion.h"
#include "MRToFromEigen.h"
#include "MRGTest.h"
#include <Eigen/Eigenvalues>

namespace MR
{

inline Matrix4d calculateMatrixP( const Matrix3d & s )
{
    return //symmetric matrix
    {
        { s.x.x + s.y.y + s.z.z,    s.y.z - s.z.y,            s.z.x - s.x.z,            s.x.y - s.y.x },
        { s.y.z - s.z.y,            s.x.x - s.y.y - s.z.z,    s.x.y + s.y.x,            s.z.x + s.x.z },
        { s.z.x - s.x.z,            s.x.y + s.y.x,            s.y.y - s.x.x - s.z.z,    s.y.z + s.z.y },
        { s.x.y - s.y.x,            s.z.x + s.x.z,            s.y.z + s.z.y,            s.z.z - s.x.x - s.y.y }
    };
}

SymMatrix3d caluclate2DimensionsP( const Matrix4d& P, const Vector4d& d1, const Vector4d& d2 )
{
    // P must be symmetric
    const auto p0 = P.col( 0 );
    const auto p1 = P * d1;
    const auto p2 = P * d2;
    SymMatrix3d res;
    res.xx = p0.x;
    res.xy = p1.x;
    res.xz = p2.x;
    res.yy = dot( d1, p1 );
    res.yz = dot( d1, p2 );
    res.zz = dot( d2, p2 );
    return res;
}

void PointToPointAligningTransform::add( const Vector3d& p1, const Vector3d& p2, double w /*= 1.0*/ )
{
    sum12_ += w * outer( p1, p2 );
    sum1_ += w * p1;
    sum2_ += w * p2;
    sumW_ += w;
}

void PointToPointAligningTransform::add( const PointToPointAligningTransform & other )
{
    sum12_ += other.sum12_;
    sum1_ += other.sum1_;
    sum2_ += other.sum2_;
    sumW_ += other.sumW_;
}

auto PointToPointAligningTransform::findPureRotation_() const -> BestRotation
{
    // for more detail of this algorithm see paragraph "3.3 A solution involving unit quaternions" in 
    // http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    Matrix3d s = sum12_ - totalWeight() * outer( centroid1(), centroid2() );
    Matrix4d p = calculateMatrixP( s );

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver( toEigen( p ) );
    Eigen::Vector4d largestEigenVector = solver.eigenvectors().col( 3 );
    Quaterniond q( largestEigenVector[0], largestEigenVector[1], largestEigenVector[2], largestEigenVector[3] );
    return { Matrix3d{ q }, solver.eigenvalues()( 3 ) };
}

AffineXf3d PointToPointAligningTransform::findBestRigidXf() const
{
    const Matrix3d r = findPureRotation_().rot;
    const auto shift = centroid2() - r * centroid1();
    return AffineXf3d( r, shift );
}

AffineXf3d PointToPointAligningTransform::findBestRigidXfFixedRotationAxis( const Vector3d & axis ) const
{
    if ( axis.lengthSq() <= 0 )
        return findBestRigidXf();

    const auto centroid1 = this->centroid1();
    const auto centroid2 = this->centroid2();
    const auto totalWeight = this->totalWeight();

    const Matrix3d s = sum12_ - totalWeight * outer( centroid1, centroid2 );
    const auto k = axis.normalized();

    // a = sum_i( dot( p2_i, cross( k, cross( k, p1_i ) ) )
    const auto a =
        ( k.x * k.x - 1 ) * s.x.x +
        ( k.y * k.y - 1 ) * s.y.y +
        ( k.z * k.z - 1 ) * s.z.z +
        ( k.x * k.y ) * ( s.x.y + s.y.x ) +
        ( k.x * k.z ) * ( s.x.z + s.z.x ) +
        ( k.y * k.z ) * ( s.y.z + s.z.y );

    // b = dot( k, sum_i cross( p1_i, p2_i ) )
    const auto b = 
        k.x * ( s.y.z - s.z.y ) +
        k.y * ( s.z.x - s.x.z ) +
        k.z * ( s.x.y - s.y.x );

    const auto phi = atan2( b, -a );

    const auto r = Matrix3d::rotation( k, phi );
    const auto shift = centroid2 - r * centroid1;
    return AffineXf3d( r, shift );
}

AffineXf3d PointToPointAligningTransform::findBestRigidXfOrthogonalRotationAxis( const Vector3d& ort ) const
{
    // for more detail of this algorithm see paragraph "3.3 A solution involving unit quaternions" in 
    // http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    const auto centroid1 = this->centroid1();
    const auto centroid2 = this->centroid2();
    const auto totalWeight = this->totalWeight();

    const Matrix3d s = sum12_ - totalWeight * outer( centroid1, centroid2 );
    const Matrix4d p = calculateMatrixP( s );

    const auto [d1, d2] = ort.perpendicular();
    const SymMatrix3d p2d = caluclate2DimensionsP( p, Vector4d{0,d1[0],d1[1],d1[2]}, Vector4d{0,d2[0],d2[1],d2[2]} );

    Matrix3d eigenvectors;
    p2d.eigens( &eigenvectors );
    const Vector3d largestEigenVector = eigenvectors.z;
    const Quaterniond q =
        Quaterniond{ 1,     0,     0,     0 } * largestEigenVector.x +
        Quaterniond{ 0, d1[0], d1[1], d1[2] } * largestEigenVector.y +
        Quaterniond{ 0, d2[0], d2[1], d2[2] } * largestEigenVector.z;

    const Matrix3d r{ q };
    const auto shift = centroid2 - r * centroid1;
    return AffineXf3d( r, shift );
}

Vector3d PointToPointAligningTransform::findBestTranslation() const
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
    auto xf1 = at1.findBestRigidXf();

    ASSERT_NEAR( ( xf1( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );

    auto xf1_ = at1.findBestRigidXfFixedRotationAxis( Vector3d{ 1, 1, 1 } );
    ASSERT_NEAR( ( xf1_( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1_( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1_( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );


    auto xf1__ = at1.findBestRigidXfOrthogonalRotationAxis( Vector3d{1, 0, -1} );
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

    auto xf2 = at2.findBestRigidXf();
    EXPECT_NEAR( ( xf2( p11 ) - p12 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p21 ) - p22 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p31 ) - p32 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p41 ) - p42 ).length(), 0., 1e-14 );
    EXPECT_NEAR( ( xf2( p51 ) - p52 ).length(), 0., 1e-14 );

    auto xf2_ = at2.findBestRigidXfFixedRotationAxis( Vector3d{ -1, 1, -1 } );
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

    auto xf3 = at3.findBestRigidXf();
    EXPECT_NEAR( ( xf3( p11 ) - p12 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p21 ) - p22 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p31 ) - p32 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p41 ) - p42 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3( p51 ) - p52 ).length(), 0., 1e-13 );

    auto xf3_ = at3.findBestRigidXfFixedRotationAxis( Vector3d{ 1, 1, 1 } );
    EXPECT_NEAR( ( xf3_( p11 ) - p12 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p21 ) - p22 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p31 ) - p32 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p41 ) - p42 ).length(), 0., 1e-13 );
    EXPECT_NEAR( ( xf3_( p51 ) - p52 ).length(), 0., 1e-13 );
}

} //namespace MR
