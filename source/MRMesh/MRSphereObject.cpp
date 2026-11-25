#include "MRSphereObject.h"
#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRObjectFactory.h"
#include "MRIRenderObject.h"
#include "MRSymMatrix3.h"
#include <MRPch/MRJson.h>

namespace MR
{

MR_ADD_CLASS_FACTORY( SphereObject )

float SphereObject::getRadius( ViewportId id /*= {}*/ ) const
{
    return s_.get( id ).x.x;
}

Vector3f SphereObject::getCenter( ViewportId id /*= {}*/ ) const
{
    return xf( id ).b;
}

void SphereObject::setRadius( float radius, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.A = Matrix3f::scale( radius );
    setXf( currentXf, id );
}

void SphereObject::setCenter( const Vector3f& center, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.b = center;
    setXf( currentXf, id );
}


const std::vector<FeatureObjectSharedProperty>& SphereObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Radius", FeaturePropertyKind::linearDimension, &SphereObject::getRadius, &SphereObject::setRadius},
       {"Center", FeaturePropertyKind::position,        &SphereObject::getCenter, &SphereObject::setCenter}
    };
    return ret;
}

FeatureObjectProjectPointResult SphereObject::projectPoint( const Vector3f& point, ViewportId id /*= {}*/ ) const
{
    const Vector3f& center = getCenter( id );
    const float radius = getRadius( id );

    auto X = point - center;
    auto normal = X.normalized();

    auto projection = center + normal * radius;
    return { projection, normal };
}

SphereObject::SphereObject()
    : AddVisualProperties( 2 )
{}

SphereObject::SphereObject( const std::vector<Vector3f>& pointsToApprox )
    : SphereObject()
{
    // least squares of ( |p - c|^2 - r^2 ), translation-invariant: relative to the centroid m of the points,
    // it gives S * ( c - m ) = t / 2 for the scatter matrix S, and r^2 = |c - m|^2 + mean |p - m|^2;
    // the sums are taken relative to the first point and then moved to the centroid
    if ( pointsToApprox.empty() )
    {
        setCenter( Vector3f{} );
        setRadius( 0.0f );
        return;
    }
    const Vector3d p0( pointsToApprox[0] );
    SymMatrix3d syy;
    Vector3d sy, ry;
    double q = 0;
    for ( const auto& pta : pointsToApprox )
    {
        const Vector3d y = Vector3d( pta ) - p0;
        const double yy = y.lengthSq();
        sy += y;
        syy += outerSquare( y );
        ry += yy * y;
        q += yy;
    }
    const double n = double( pointsToApprox.size() );
    const Vector3d m = sy / n;
    const double mm = m.lengthSq();
    const SymMatrix3d s = syy - outerSquare( n, m );
    const Vector3d t = ry - 2.0 * ( syy * m ) - q * m + ( 2 * n * mm ) * m;
    const Vector3d c = 0.5 * ( s.pseudoinverse() * t );
    setCenter( Vector3f( p0 + m + c ) );
    setRadius( float( std::sqrt( std::max( c.lengthSq() + q / n - mm, 0.0 ) ) ) );
}

std::shared_ptr<Object> SphereObject::shallowClone() const
{
    return std::make_shared<SphereObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> SphereObject::clone() const
{
    return std::make_shared<SphereObject>( ProtectedStruct{}, *this );
}

void SphereObject::swapBase_( Object& other )
{
    if ( auto sphereObject = other.asType<SphereObject>() )
        std::swap( *this, *sphereObject );
    else
        assert( false );
}

void SphereObject::serializeFields_( Json::Value& root ) const
{
    FeatureObject::serializeFields_( root );
    root["Type"].append( SphereObject::StaticTypeName() );
}

void SphereObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

}
