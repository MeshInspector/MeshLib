#include "MRPointObject.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRVector3.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( PointObject )

PointObject::PointObject()
    : FeatureObject( 0 )
{}

PointObject::PointObject( const std::vector<Vector3f>& pointsToApprox )
    : PointObject()
{
    Vector3d center;
    for ( auto& p : pointsToApprox )
        center += Vector3d( p );
    setPoint( Vector3f( center / double( pointsToApprox.size() ) ) );
}

std::shared_ptr<MR::Object> PointObject::clone() const
{
    return std::make_shared<PointObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<MR::Object> PointObject::shallowClone() const
{
    return std::make_shared<PointObject>( ProtectedStruct{}, *this );
}

Vector3f PointObject::getPoint( ViewportId id /*= {}*/ ) const
{
    return xf( id ).b;
}

void PointObject::setPoint( const Vector3f& point, ViewportId id /*= {}*/ )
{
    setXf( AffineXf3f::translation( point ), id );
}

std::vector<FeatureObjectSharedProperty>& PointObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Point", FeaturePropertyKind::position, &PointObject::getPoint, &PointObject::setPoint}
    };
    return ret;
}

FeatureObjectProjectPointResult PointObject::projectPoint( const Vector3f& /*point*/, ViewportId id /*= {}*/ ) const
{
    return { getPoint( id ) , std::nullopt };
}

void PointObject::swapBase_( Object& other )
{
    if ( auto pointObject = other.asType<PointObject>() )
        std::swap( *this, *pointObject );
    else
        assert( false );
}

void PointObject::serializeFields_( Json::Value& root ) const
{
    FeatureObject::serializeFields_( root );
    root["Type"].append( PointObject::TypeName() );
}

void PointObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

}
