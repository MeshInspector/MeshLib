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
    currentXf.A = Matrix3f::rotation( Vector3f::plusX(), normal ) * Matrix3f::scale( currentXf.A.toScale() );
    setXf( currentXf );
}

void LineObject::setCenter( const Vector3f& center )
{
    auto currentXf = xf();
    currentXf.b = center;
    setXf( currentXf );
}

void LineObject::setSize( float size )
{
    auto currentXf = xf();
    currentXf.A = Matrix3f::rotationFromEuler( currentXf.A.toEulerAngles() ) * Matrix3f::scale( Vector3f::diagonal( size ) );
    setXf( currentXf );
}

LineObject::LineObject()
{
    constructPolyline_();
}

LineObject::LineObject( const std::vector<Vector3f>& pointsToApprox )
{
    constructPolyline_();

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
    setSize( box.diagonal() * 2.f );
}

std::vector<std::string> LineObject::getInfoLines() const
{
    std::vector<std::string> res;

    res.push_back( "type : LineObject" );
    return res;
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
    if ( auto planeObject = other.asType<LineObject>() )
        std::swap( *this, *planeObject );
    else
        assert( false );
}

void LineObject::serializeFields_( Json::Value& root ) const
{
    ObjectLinesHolder::serializeFields_( root );
    root["Type"].append( LineObject::TypeName() );
}

void LineObject::constructPolyline_()
{
    // create object Polyline
    Polyline3 lineObj;
    const std::vector<Vector3f> points = { Vector3f::minusX(), Vector3f::plusX() };
    lineObj.addFromPoints( points.data(), 2 );

    polyline_ = std::make_shared<Polyline3>( lineObj );

    setDirtyFlags( DIRTY_ALL );
}

}
