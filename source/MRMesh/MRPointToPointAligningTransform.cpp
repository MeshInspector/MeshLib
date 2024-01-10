#include "MRPointToPointAligningTransform.h"
#include "MRVector3.h"
#include "MRSymMatrix3.h"
#include "MRSymMatrix4.h"
#include "MRQuaternion.h"
#include "MRToFromEigen.h"
#include "MRGTest.h"
#include <Eigen/Eigenvalues>

namespace MR
{

inline SymMatrix4d calculateMatrixP( const Matrix3d & s )
{
    SymMatrix4d P;
    P.xx = s.x.x + s.y.y + s.z.z;  P.xy = s.y.z - s.z.y;          P.xz = s.z.x - s.x.z;          P.xw = s.x.y - s.y.x;
                                   P.yy = s.x.x - s.y.y - s.z.z;  P.yz = s.x.y + s.y.x,          P.yw = s.z.x + s.x.z;
                                                                  P.zz = s.y.y - s.x.x - s.z.z;  P.zw = s.y.z + s.z.y;
                                                                                                 P.ww = s.z.z - s.x.x - s.y.y;
    return P;
}

SymMatrix3d caluclate2DimensionsP( const SymMatrix4d& P, const Vector4d& d1, const Vector4d& d2 )
{
    const Vector4d p0{ P.xx, P.xy, P.xz, P.xw };
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
    sum11_ += w * p1.lengthSq();
    sumW_ += w;
}

void PointToPointAligningTransform::add( const PointToPointAligningTransform & other )
{
    sum12_ += other.sum12_;
    sum1_ += other.sum1_;
    sum2_ += other.sum2_;
    sum11_ += other.sum11_;
    sumW_ += other.sumW_;
}

auto PointToPointAligningTransform::findPureRotation_() const -> BestRotation
{
    assert( totalWeight() > 0 );

    // for more detail of this algorithm see paragraph "3.3 A solution involving unit quaternions" in 
    // http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    const Matrix3d s = sum12_ - outer( sum1_, centroid2() );
    const SymMatrix4d p = calculateMatrixP( s );

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver( toEigen( p ) );
    Eigen::Vector4d largestEigenVector = solver.eigenvectors().col( 3 );
    Quaterniond q( largestEigenVector[0], largestEigenVector[1], largestEigenVector[2], largestEigenVector[3] );
    return { Matrix3d{ q }, solver.eigenvalues()( 3 ) };
}

AffineXf3d PointToPointAligningTransform::findBestRigidXf() const
{
    if ( totalWeight() <= 0 )
        return {};
    const Matrix3d r = findPureRotation_().rot;
    const auto shift = centroid2() - r * centroid1();
    return AffineXf3d( r, shift );
}

AffineXf3d PointToPointAligningTransform::findBestRigidScaleXf() const
{
    if ( totalWeight() <= 0 )
        return {};
    const auto x = findPureRotation_();

    const double dev11 = sum11_ - sum1_.lengthSq() / totalWeight();
    assert( x.err > 0 );
    assert( dev11 > 0 );
    const auto scale = x.err / dev11;

    const Matrix3d m = x.rot * scale;
    const auto shift = centroid2() - m * centroid1();
    return AffineXf3d( m, shift );
}

AffineXf3d PointToPointAligningTransform::findBestRigidXfFixedRotationAxis( const Vector3d & axis ) const
{
    if ( axis.lengthSq() <= 0 )
        return findBestRigidXf();

    const auto centroid1 = this->centroid1();
    const auto centroid2 = this->centroid2();

    const Matrix3d s = sum12_ - outer( sum1_, centroid2 );
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

    const Matrix3d s = sum12_ - outer( sum1_, centroid2 );
    const SymMatrix4d p = calculateMatrixP( s );

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


TEST( MRMesh, PointToPointAligningTransform1 )
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
 
    xf1 = at1.findBestRigidScaleXf();
    ASSERT_NEAR( ( xf1( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );

    xf1 = at1.findBestRigidXfFixedRotationAxis( Vector3d{ 1, 1, 1 } );
    ASSERT_NEAR( ( xf1( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );

    xf1 = at1.findBestRigidXfOrthogonalRotationAxis( Vector3d{1, 0, -1} );
    ASSERT_NEAR( ( xf1( Vector3d::plusX() ) - Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusY() ) - Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf1( Vector3d::plusZ() ) - Vector3d::plusX() - b ).length(), 0., 1e-15 );

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


    PointToPointAligningTransform at4;
    double scale = 0.25;
    at4.add( Vector3d::plusX(), scale * Vector3d::plusY() + b );
    at4.add( Vector3d::plusY(), scale * Vector3d::plusZ() + b );
    at4.add( Vector3d::plusZ(), scale * Vector3d::plusX() + b );

    auto xf4 = at4.findBestRigidScaleXf();
    ASSERT_NEAR( ( xf4( Vector3d::plusX() ) - scale * Vector3d::plusY() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf4( Vector3d::plusY() ) - scale * Vector3d::plusZ() - b ).length(), 0., 1e-15 );
    ASSERT_NEAR( ( xf4( Vector3d::plusZ() ) - scale * Vector3d::plusX() - b ).length(), 0., 1e-15 );
}

TEST(MRMesh, PointToPointAligningTransform2 )
{
    // set points
    const std::vector<Vector3d> points = {
        {   1.0,   1.0, -5.0 },
        {  14.0,   1.0,  1.0 },
        {   1.0,  14.0,  2.0 },
        { -11.0,   2.0,  3.0 },
        {   1.0, -11.0,  4.0 },
        {   1.0,   2.0,  8.0 },
        {   2.0,   1.0, -5.0 },
        {  15.0,   1.5,  1.0 },
        {   1.5,  15.0,  2.0 },
        { -11.0,   2.5,  3.1 },
    };
    // Point to Point part
    const std::vector<AffineXf3d> xfs = {
        // zero xf
        AffineXf3d(
            Matrix3d(
                Vector3d(1, 0, 0),
                Vector3d(0, 1, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(0,0,0)),

        // small Rz
        AffineXf3d(
            Matrix3d(
                Vector3d(0.8, 0.6, 0),
                Vector3d(-0.6, 0.8, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(0,0,0)),

        // small transl
        AffineXf3d(
            Matrix3d(
                Vector3d(0.8, 0.6, 0),
                Vector3d(-0.6, 0.8, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(2,-2,0)),

        // complex xf
        AffineXf3d(
            Matrix3d(
                Vector3d(0.8, 0, -0.6),
                Vector3d(0, 1, 0),
                Vector3d(0.6, 0, 0.8)
            ),
            Vector3d(200,-200,0)),
    };

    constexpr auto eps = 5e-14;
    for ( const auto& xf : xfs )
    {
        {
            PointToPointAligningTransform p2pt;
            for ( const auto & p : points )
                p2pt.add( p, xf( p ) );

            auto xfResP2pt = p2pt.findBestRigidXf();
            EXPECT_NEAR((xfResP2pt.A - xf.A).norm(), 0., eps);
            EXPECT_NEAR((xfResP2pt.b - xf.b).length(), 0., eps);
        }
        {
            auto scaleXf = xf;
            scaleXf.A *= 3.0;

            PointToPointAligningTransform p2ptS;
            for ( const auto & p : points )
                p2ptS.add( p, scaleXf( p ) );

            auto xfResP2ptS = p2ptS.findBestRigidScaleXf();
            EXPECT_NEAR((xfResP2ptS.A - scaleXf.A).norm(), 0., eps);
            EXPECT_NEAR((xfResP2ptS.b - scaleXf.b).length(), 0., eps);
        }
    }
}

} //namespace MR
