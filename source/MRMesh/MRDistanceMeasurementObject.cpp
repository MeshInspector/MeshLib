#include "MRDistanceMeasurementObject.h"

#include "MRMesh/MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRFmt.h"

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
    if ( drawAsNegative_ != value )
    {
        drawAsNegative_ = value;
        cachedValue_ = {};
    }
}

DistanceMeasurementObject::PerCoordDeltas DistanceMeasurementObject::getPerCoordDeltasMode() const
{
    return perCoordDeltas_;
}

void DistanceMeasurementObject::setPerCoordDeltasMode( PerCoordDeltas mode )
{
    perCoordDeltas_ = mode;
}

float DistanceMeasurementObject::computeDistance() const
{
    if ( !cachedValue_ )
        cachedValue_ = getWorldDelta().length() * ( getDrawAsNegative() ? -1.f : 1.f );
    return *cachedValue_;
}

std::vector<std::string> DistanceMeasurementObject::getInfoLines() const
{
    auto ret = MeasurementObject::getInfoLines();
    ret.push_back( fmt::format( "distance value: {:.3f}", computeDistance() ) );
    return ret;
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

void DistanceMeasurementObject::propagateWorldXfChangedSignal_()
{
    MeasurementObject::propagateWorldXfChangedSignal_();
    cachedValue_ = {};
}

} // namespace MR
