#include "MRDistanceMeasurementObject.h"

#include "MRMesh/MRObjectFactory.h"
#include "MRPch/MRJson.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( DistanceMeasurementObject )

std::shared_ptr<Object> DistanceMeasurementObject::clone() const
{
    return std::make_shared<DistanceMeasurementObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> DistanceMeasurementObject::shallowClone() const
{
    return DistanceMeasurementObject::clone();
}

Vector3f DistanceMeasurementObject::getWorldPoint() const
{
    Vector3f ret = getLocalPoint();
    if ( parent() )
        ret = parent()->worldXf()( ret );
    return ret;
}

Vector3f DistanceMeasurementObject::getLocalPoint() const
{
    return xf().b;
}

Vector3f DistanceMeasurementObject::getWorldDelta() const
{
    Vector3f ret = getLocalDelta();
    if ( parent() )
        ret = parent()->worldXf().A * ret;
    return ret;
}

Vector3f DistanceMeasurementObject::getLocalDelta() const
{
    return xf().A.col( 0 );
}

void DistanceMeasurementObject::setLocalPoint( const MR::Vector3f& point )
{
    auto curXf = xf();
    curXf.b = point;
    setXf( curXf );
}

void DistanceMeasurementObject::setLocalDelta( const MR::Vector3f& delta )
{
    auto curXf = xf();
    auto basis = delta.perpendicular();
    curXf.A = Matrix3f( delta, basis.first, basis.second ).transposed();
    setXf( curXf );
}

bool DistanceMeasurementObject::getDrawAsNegative() const
{
    return drawAsNegative_;
}

void DistanceMeasurementObject::setDrawAsNegative( bool value )
{
    drawAsNegative_ = value;
}

DistanceMeasurementObject::PerCoordDeltas DistanceMeasurementObject::getPerCoordDeltasMode() const
{
    return perCoordDeltas_;
}

void DistanceMeasurementObject::setPerCoordDeltasMode( PerCoordDeltas mode )
{
    perCoordDeltas_ = mode;
}

void DistanceMeasurementObject::swapBase_( Object& other )
{
    if ( auto ptr = other.asType<DistanceMeasurementObject>() )
        std::swap( *this, *ptr );
    else
        assert( false );
}

void DistanceMeasurementObject::serializeFields_( Json::Value& root ) const
{
    MeasurementObject::serializeFields_( root );
    root["Type"].append( TypeName() );

    root["DrawAsNegative"] = drawAsNegative_;
}

void DistanceMeasurementObject::deserializeFields_( const Json::Value& root )
{
    MeasurementObject::deserializeFields_( root );

    if ( const auto& json = root["DrawAsNegative"]; json.isBool() )
        drawAsNegative_ = json.asBool();

    if ( const auto& json = root["PerCoordDeltas"]; json.isInt() )
        perCoordDeltas_ = PerCoordDeltas( json.asInt() );
}

void DistanceMeasurementObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

} // namespace MR
