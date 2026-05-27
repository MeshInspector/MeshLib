#include "MRPointToPointAligningTransform.h"
#include "MRVector3.h"
#include "MRSymMatrix3.h"
#include "MRSymMatrix4.h"
#include "MRQuaternion.h"
#include "MRToFromEigen.h"
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

} //namespace MR
