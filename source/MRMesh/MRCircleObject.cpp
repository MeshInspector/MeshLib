#include "MRCircleObject.h"
#include "MRMatrix3.h"
#include "MRPolyline.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include <Eigen/Dense>
#include "MRConstants.h"
#include "MRBestFit.h"

namespace
{
constexpr int cDetailLevel = 128;
}

namespace MR
{

MR_ADD_CLASS_FACTORY( CircleObject )

float CircleObject::getRadius() const
{
    return xf().A.toScale().x;
}

Vector3f CircleObject::getCenter() const
{
    return xf().b;
}

Vector3f CircleObject::getNormal() const
{
    return ( xf().A * Vector3f::plusZ() ).normalized();
}

void CircleObject::setRadius( float radius )
{
    auto currentXf = xf();
    currentXf.A = Matrix3f::rotationFromEuler( currentXf.A.toEulerAngles() ) * Matrix3f::scale( radius );
    setXf( currentXf );
}

void CircleObject::setCenter( const Vector3f& center )
{
    auto currentXf = xf();
    currentXf.b = center;
    setXf( currentXf );
}

void CircleObject::setNormal( const Vector3f& normal )
{
    auto currentXf = xf();
    currentXf.A = Matrix3f::rotation( Vector3f::plusZ(), normal ) * Matrix3f::scale( currentXf.A.toScale() );
    setXf( currentXf );
}

CircleObject::CircleObject()
{
    constructPolyline_();
}

CircleObject::CircleObject( const std::vector<Vector3f>& pointsToApprox )
{
    constructPolyline_();

    PointAccumulator pa;
    for ( const auto& p : pointsToApprox )
        pa.addPoint( p );

    // make a normal vector from center directed against a point (0, 0, 0)
    Plane3f plane = pa.getBestPlanef();
    Vector3f normal = plane.n.normalized();
    if ( plane.d < 0 )
        normal *= -1.f;

    AffineXf3f toPlaneXf = AffineXf3f( Matrix3f::rotation( Vector3f::plusZ(), normal ), plane.n * plane.d ).inverse();

    std::vector<Vector3f> pointsProj( pointsToApprox.size() );
    for ( int i = 0; i < pointsProj.size(); ++i )
        pointsProj[i] = toPlaneXf( plane.project( pointsToApprox[i] ) );


    // find best radius and center
    Eigen::Matrix<double, 3, 3> accumA_;
    Eigen::Matrix<double, 3, 1> accumB_;
    accumA_.setZero();
    accumB_.setZero();
    for ( const auto& pta : pointsProj )
    {
        Eigen::Matrix<double, 3, 1> vec;
        vec[0] = 2.0 * pta.x;
        vec[1] = 2.0 * pta.y;
        vec[2] = -1.0;

        accumA_ += ( vec * vec.transpose() );
        accumB_ += ( vec * ( pta.x * pta.x + pta.y * pta.y ) );
    }
    Eigen::Matrix<double, 3, 1> res = Eigen::Matrix<double, 3, 1>( accumA_.colPivHouseholderQr().solve( accumB_ ) );
    Vector3f center{ float( res[0] ), float( res[1] ), 0.f };
    double rSq = res[0] * res[0] + res[1] * res[1] - res[2];
    assert( rSq >= 0.0 );
    float radius = float( sqrt( std::max( rSq, 0.0 ) ) );

    setNormal( normal );
    setCenter( toPlaneXf.inverse()( center ) );
    setRadius( radius );
}

std::shared_ptr<Object> CircleObject::shallowClone() const
{
    auto res = std::make_shared<CircleObject>( ProtectedStruct{}, *this );
    if ( polyline_ )
        res->polyline_ = polyline_;
    return res;
}

std::shared_ptr<Object> CircleObject::clone() const
{
    auto res = std::make_shared<CircleObject>( ProtectedStruct{}, *this );
    if ( polyline_ )
        res->polyline_ = std::make_shared<Polyline3>( *polyline_ );
    return res;
}

void CircleObject::swapBase_( Object& other )
{
    if ( auto sphereObject = other.asType<CircleObject>() )
        std::swap( *this, *sphereObject );
    else
        assert( false );
}

void CircleObject::serializeFields_( Json::Value& root ) const
{
    ObjectLinesHolder::serializeFields_( root );
    root["Type"].append( CircleObject::TypeName() );
}

void CircleObject::constructPolyline_()
{
    polyline_ = std::make_shared<Polyline3>();

    std::vector<Vector3f> points( cDetailLevel );
    for ( int i = 0; i < cDetailLevel; ++i )
    {
        points[i].x = cosf( i / 32.f * PI_F );
        points[i].y = sinf( i / 32.f * PI_F );
    }
    polyline_->addFromPoints( points.data(), cDetailLevel, true );

    setDirtyFlags( DIRTY_ALL );
}

}