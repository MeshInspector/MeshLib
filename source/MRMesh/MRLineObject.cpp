#include "MRLineObject.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRBestFit.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRMatrix3.h"
#include "MRVector3.h"
#include "MRMatrix3Decompose.h"

namespace MR
{

// default length of line. Historically it eq. 2 (but 1 looks better). Left as is for compatibility.
size_t baseLineObjectLength_ = 2;

MR_ADD_CLASS_FACTORY( LineObject )

Vector3f LineObject::getDirection() const
{
    return ( xf().A * Vector3f::plusX() ).normalized();
}

Vector3f LineObject::getCenter() const
{
    return xf().b;
}

void LineObject::setDirection( const Vector3f& normal )
{
    auto currentXf = xf();
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    currentXf.A = Matrix3f::rotation( Vector3f::plusX(), normal ) * s;
    setXf( currentXf );
}

void LineObject::setCenter( const Vector3f& center )
{
    auto currentXf = xf();
    currentXf.b = center;
    setXf( currentXf );
}

void LineObject::setLength( float size )
{
    auto currentXf = xf();
    currentXf.A = Matrix3f::rotationFromEuler( currentXf.A.toEulerAngles() ) * Matrix3f::scale( Vector3f::diagonal( size / baseLineObjectLength_ ) );
    setXf( currentXf );
}

float LineObject::getLength() const
{
    Matrix3f r, s;
    decomposeMatrix3( xf().A, r, s );
    return s.x.x * baseLineObjectLength_;
}

Vector3f LineObject::getPointA() const
{
    return getCenter() - getDirection() * ( getLength() / 2 );
}

Vector3f LineObject::getPointB() const
{
    return getCenter() + getDirection() * ( getLength() / 2 );
}


LineObject::LineObject()
{
    constructPolyline_();
}

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
    setLength( box.diagonal() * 4 );
}

std::shared_ptr<Object> LineObject::shallowClone() const
{
    auto res = std::make_shared<LineObject>( ProtectedStruct{}, *this );
    if ( polyline_ )
        res->polyline_ = polyline_;
    return res;
}

std::shared_ptr<Object> LineObject::clone() const
{
    auto res = std::make_shared<LineObject>( ProtectedStruct{}, *this );
    if ( polyline_ )
        res->polyline_ = std::make_shared<Polyline3>( *polyline_ );
    return res;
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
    ObjectLinesHolder::serializeFields_( root );
    root["Type"].append( LineObject::TypeName() );
}

void LineObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype(*this)>( *this );
}

void LineObject::constructPolyline_()
{
    // create object Polyline
    Polyline3 lineObj;
    const std::vector<Vector3f> points = { Vector3f::minusX(), Vector3f::plusX() };
    lineObj.addFromPoints( points.data(), baseLineObjectLength_ );

    polyline_ = std::make_shared<Polyline3>( lineObj );

    setDirtyFlags( DIRTY_ALL );
}

const std::vector<FeatureObjectSharedProperty>& LineObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Center"   , &LineObject::getCenter   , &LineObject::setCenter},
       {"Direction", &LineObject::getDirection, &LineObject::setDirection},
       {"Length"   , &LineObject::getLength   , &LineObject::setLength}
    };
    return ret;
}

}
