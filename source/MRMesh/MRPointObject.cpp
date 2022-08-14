#include "MRPointObject.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRVector3.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( PointObject )

PointObject::PointObject()
{
    constructPointCloud_();
}

PointObject::PointObject( const std::vector<Vector3f>& pointsToApprox )
{
    constructPointCloud_();
    Vector3d center;
    for ( auto& p : pointsToApprox )
        center += Vector3d( p );
    setPoint( Vector3f( center / double( pointsToApprox.size() ) ) );
}

std::shared_ptr<MR::Object> PointObject::clone() const
{
    auto res = std::make_shared<PointObject>( ProtectedStruct{}, *this );
    if ( points_ )
        res->points_ = std::make_shared<PointCloud>( *points_ );
    return res;
}

std::shared_ptr<MR::Object> PointObject::shallowClone() const
{
    auto res = std::make_shared<PointObject>( ProtectedStruct{}, *this );
    if ( points_ )
        res->points_ = points_;
    return res;
}

Vector3f PointObject::getPoint() const
{
    return xf().b;
}

void PointObject::setPoint( const Vector3f& point )
{
    setXf( AffineXf3f::translation( point ) );
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
    ObjectPointsHolder::serializeFields_( root );
    root["Type"].append( PointObject::TypeName() );
}

void PointObject::constructPointCloud_()
{
    points_ = std::make_shared<PointCloud>();
    points_->points.push_back( Vector3f() );
    points_->validPoints.resize( 1, true );

    setDirtyFlags( DIRTY_ALL );
}

}
