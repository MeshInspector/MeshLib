#include "MRLineObject.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRBestFit.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRMatrix3.h"
#include "MRVector3.h"

namespace MR
{

// default length of line. Historically it eq. 2 (but 1 looks better). Left as is for compatibility.
size_t baseLineObjectLength_ = 2;

MR_ADD_CLASS_FACTORY( LineObject )

Vector3f LineObject::getDirection( ViewportId id /*= {}*/ ) const
{
    return ( xf( id ).A * Vector3f::plusX() ).normalized();
}

Vector3f LineObject::getCenter( ViewportId id /*= {}*/ ) const
{
    return xf( id ).b;
}

void LineObject::setDirection( const Vector3f& normal, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.A = Matrix3f::rotation( Vector3f::plusX(), normal ) * s_.get( id );
    setXf( currentXf );
}

void LineObject::setCenter( const Vector3f& center, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.b = center;
    setXf( currentXf, id );
}

void LineObject::setLength( float size, ViewportId id /*= {}*/ )
{
    auto currentXf = xf( id );
    currentXf.A = Matrix3f::rotationFromEuler( currentXf.A.toEulerAngles() ) * Matrix3f::scale( Vector3f::diagonal( size / baseLineObjectLength_ ) );
    setXf( currentXf, id );
}

float LineObject::getLength( ViewportId id /*= {}*/ ) const
{
    return s_.get( id ).x.x * baseLineObjectLength_;
}

Vector3f LineObject::getBasePoint( ViewportId id /*= {} */ ) const
{
    return getPointA( id );
}

Vector3f LineObject::getPointA( ViewportId id /*= {}*/ ) const
{
    return getCenter( id ) - getDirection( id ) * ( getLength( id ) / 2 );
}

Vector3f LineObject::getPointB( ViewportId id /*= {}*/ ) const
{
    return getCenter( id ) + getDirection( id ) * ( getLength( id ) / 2 );
}


LineObject::LineObject()
    : FeatureObject( 1 )
{}

LineObject::LineObject( const std::vector<Vector3f>& pointsToApprox )
    : LineObject()
{
    PointAccumulator pa;
    Box3f box;
    for ( const auto& p : pointsToApprox )
    {
        pa.addPoint( p );
        box.include( p );
    }

    // make a normal vector from center directed against a point (0, 0, 0)
    Line3f line = pa.getBestLinef();
    Vector3f dir = line.d.normalized();
    const Vector3f bboxCenterProj = line.project( box.center() );
    if ( ( bboxCenterProj + dir ).lengthSq() < bboxCenterProj.lengthSq() )
        dir *= -1.f;

    setDirection( dir );
    setCenter( box.center() );
    setLength( box.diagonal() );
}

std::shared_ptr<Object> LineObject::shallowClone() const
{
    return std::make_shared<LineObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> LineObject::clone() const
{
    return std::make_shared<LineObject>( ProtectedStruct{}, *this );
}

void LineObject::swapBase_( Object& other )
{
    if ( auto lineObject = other.asType<LineObject>() )
        std::swap( *this, *lineObject );
    else
        assert( false );
}

void LineObject::serializeFields_( Json::Value& root ) const
{
    FeatureObject::serializeFields_( root );
    root["Type"].append( LineObject::TypeName() );
}

void LineObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

FeatureObjectProjectPointResult LineObject::projectPoint( const Vector3f& point, ViewportId id /*= {}*/ ) const
{
    const Vector3f& center = getCenter( id );
    const Vector3f& direction = getDirection( id );

    auto X = point - center;
    auto K = direction * dot( X, direction );

    return { K + center , std::nullopt };
}

const std::vector<FeatureObjectSharedProperty>& LineObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Center"   , FeaturePropertyKind::position,        &LineObject::getCenter   , &LineObject::setCenter},
       {"Direction", FeaturePropertyKind::direction,       &LineObject::getDirection, &LineObject::setDirection},
       {"Length"   , FeaturePropertyKind::linearDimension, &LineObject::getLength   , &LineObject::setLength}
    };
    return ret;
}

}
