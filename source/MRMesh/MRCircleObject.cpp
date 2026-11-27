#include "MRCircleObject.h"
#include "MRMatrix3.h"
#include "MRPolyline.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRConstants.h"
#include "MRBestFit.h"
#include "MRVector.h"
#include "MRIRenderObject.h"

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

    // the best plane passes through the centroid of the points;
    // make a normal vector from center directed against a point (0, 0, 0)
    Vector3d centroid, eigenvalues;
    Matrix3d eigenvectors;
    pa.getCenteredCovarianceEigen( centroid, eigenvectors, eigenvalues );
    const Vector3d normal = dot( eigenvectors.x, centroid ) < 0 ? -eigenvectors.x : eigenvectors.x;

    // the points relative to the centroid in an orthonormal basis of the plane
    const auto [u, v] = normal.perpendicular();
    Vector<Vector2d> proj( pointsToApprox.size() );
    for ( size_t i = 0; i < proj.size(); ++i )
    {
        const Vector3d d = Vector3d( pointsToApprox[i] ) - centroid;
        proj[i] = Vector2d( dot( d, u ), dot( d, v ) );
    }
    setNormal( Vector3f( normal ) );
    if ( const auto circle = findBestCircle( proj ) )
    {
        setCenter( Vector3f( centroid + circle->center.x * u + circle->center.y * v ) );
        setRadius( float( circle->radius ) );
    }
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
    root["Type"].append( CircleObject::StaticTypeName() );
}

}
