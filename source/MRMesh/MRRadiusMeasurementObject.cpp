#include "MRRadiusMeasurementObject.h"

#include "MRMesh/MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRFmt.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( RadiusMeasurementObject )

std::shared_ptr<Object> RadiusMeasurementObject::clone() const
{
    return std::make_shared<RadiusMeasurementObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> RadiusMeasurementObject::shallowClone() const
{
    return RadiusMeasurementObject::clone();
}

Vector3f RadiusMeasurementObject::getWorldCenter() const
{
    Vector3f ret = getLocalCenter();
    if ( parent() )
        ret = parent()->worldXf()( ret );
    return ret;
}

Vector3f RadiusMeasurementObject::getLocalCenter() const
{
    return xf().b;
}

Vector3f RadiusMeasurementObject::getWorldRadiusAsVector() const
{
    Vector3f ret = getLocalRadiusAsVector();
    if ( parent() )
        ret = parent()->worldXf().A * ret;
    return ret;
}

Vector3f RadiusMeasurementObject::getLocalRadiusAsVector() const
{
    return xf().A.col( 0 );
}

Vector3f RadiusMeasurementObject::getWorldNormal() const
{
    Vector3f ret = xf().A.col( 2 );
    if ( parent() )
        ret = parent()->worldXf().A * ret;
    return ret.normalized();
}

Vector3f RadiusMeasurementObject::getLocalNormal() const
{
    // This should be already normalized, but doing it again just in case somebody messes this up.
    return xf().A.col( 2 ).normalized();
}

void RadiusMeasurementObject::setLocalCenter( const MR::Vector3f& center )
{
    auto curXf = xf();
    curXf.b = center;
    setXf( curXf );
}

void RadiusMeasurementObject::setLocalRadiusAsVector( const MR::Vector3f& vec, const Vector3f& normal )
{
    auto curXf = xf();
    Vector3f y = cross( normal, vec ).normalized();
    curXf.A = Matrix3f( vec, y, cross( vec, y ).normalized() ).transposed();
    setXf( curXf );
}

void RadiusMeasurementObject::setDrawAsDiameter( bool value )
{
    if ( drawAsDiameter_ != value )
    {
        drawAsDiameter_ = value;
        cachedValue_ = {};
    }
}

float RadiusMeasurementObject::computeRadiusOrDiameter() const
{
    if ( !cachedValue_ )
        cachedValue_ = getWorldRadiusAsVector().length() * ( getDrawAsDiameter() ? 2.f : 1.f );
    return *cachedValue_;
}

std::vector<std::string> RadiusMeasurementObject::getInfoLines() const
{
    auto ret = MeasurementObject::getInfoLines();
    ret.push_back( fmt::format( "{} value: {:.3f}", getDrawAsDiameter() ? "diameter" : "radius", computeRadiusOrDiameter() ) );
    return ret;
}

void RadiusMeasurementObject::swapBase_( Object& other )
{
    if ( auto ptr = other.asType<RadiusMeasurementObject>() )
        std::swap( *this, *ptr );
    else
        assert( false );
}

void RadiusMeasurementObject::serializeFields_( Json::Value& root ) const
{
    MeasurementObject::serializeFields_( root );
    root["Type"].append( TypeName() );

    root["DrawAsDiameter"] = drawAsDiameter_;
    root["IsSpherical"] = isSpherical_;
    root["VisualLengthMultiplier"] = visualLengthMultiplier_;
}

void RadiusMeasurementObject::deserializeFields_( const Json::Value& root )
{
    MeasurementObject::deserializeFields_( root );

    if ( const auto& json = root["DrawAsDiameter"]; json.isBool() )
        drawAsDiameter_ = json.asBool();
    if ( const auto& json = root["IsSpherical"]; json.isBool() )
        isSpherical_ = json.asBool();
    if ( const auto& json = root["VisualLengthMultiplier"]; json.isDouble() )
        visualLengthMultiplier_ = float( json.asDouble() );
}

void RadiusMeasurementObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

void RadiusMeasurementObject::propagateWorldXfChangedSignal_()
{
    MeasurementObject::propagateWorldXfChangedSignal_();
    cachedValue_ = {};
}

} // namespace MR
