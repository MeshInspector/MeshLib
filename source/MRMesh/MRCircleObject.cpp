#include "MRCircleObject.h"
#include "MRMatrix3.h"
#include "MRPolyline.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRConstants.h"
#include "MRBestFit.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:5054)  //operator '&': deprecated between enumerations of different types
#pragma warning(disable:4127)  //C4127. "Consider using 'if constexpr' statement instead"
#elif defined(__clang__)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <Eigen/Dense>

#ifdef _MSC_VER
#pragma warning(pop)
#elif defined(__clang__)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif


namespace MR
{

MR_ADD_CLASS_FACTORY( CircleObject )

float CircleObject::getRadius( ViewportId id /*= {}*/ ) const
{
    return s_.get( id ).x.x;
}

Vector3f CircleObject::getCenter( ViewportId id /*= {}*/ ) const
{
    return xf( id ).b;
}

Vector3f CircleObject::getNormal( ViewportId id /*= {}*/ ) const
{
    return ( xf( id ).A * Vector3f::plusZ() ).normalized();
}

void CircleObject::setRadius( float radius, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.A = Matrix3f::rotationFromEuler( currentXf.A.toEulerAngles() ) * Matrix3f::scale( radius );
    setXf( currentXf, id );
}

void CircleObject::setCenter( const Vector3f& center, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.b = center;
    setXf( currentXf, id );
}

void CircleObject::setNormal( const Vector3f& normal, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );

    currentXf.A = Matrix3f::rotation( Vector3f::plusZ(), normal ) * s_.get( id );
    setXf( currentXf, id );
}

FeatureObjectProjectPointResult CircleObject::projectPoint( const Vector3f& point, ViewportId id /*= {}*/ ) const
{
    const Vector3f& center = getCenter( id );
    const float radius = getRadius( id );
    const auto& normal = getNormal( id );

    Plane3f plane( normal, dot( normal, center ) );
    auto K = plane.project( point );
    auto n = ( K - center ).normalized();
    auto projection = center + n * radius;

    return { projection, std::nullopt };
}

const std::vector<FeatureObjectSharedProperty>& CircleObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
      {"Radius", FeaturePropertyKind::linearDimension, &CircleObject::getRadius, &CircleObject::setRadius},
      {"Center", FeaturePropertyKind::position,        &CircleObject::getCenter, &CircleObject::setCenter},
      {"Normal", FeaturePropertyKind::direction,       &CircleObject::getNormal, &CircleObject::setNormal}
    };
    return ret;
}

CircleObject::CircleObject()
    : AddVisualProperties( 1 )
{}

CircleObject::CircleObject( const std::vector<Vector3f>& pointsToApprox )
    : CircleObject()
{
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
    return std::make_shared<CircleObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> CircleObject::clone() const
{
    return std::make_shared<CircleObject>( ProtectedStruct{}, *this );
}

void CircleObject::swapBase_( Object& other )
{
    if ( auto sphereObject = other.asType<CircleObject>() )
        std::swap( *this, *sphereObject );
    else
        assert( false );
}

void CircleObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

void CircleObject::serializeFields_( Json::Value& root ) const
{
    FeatureObject::serializeFields_( root );
    root["Type"].append( CircleObject::TypeName() );
}

}
