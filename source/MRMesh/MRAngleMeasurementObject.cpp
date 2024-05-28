#include "MRAngleMeasurementObject.h"

#include "MRMesh/MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRFmt.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( AngleMeasurementObject )

std::shared_ptr<Object> AngleMeasurementObject::clone() const
{
    return std::make_shared<AngleMeasurementObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> AngleMeasurementObject::shallowClone() const
{
    return AngleMeasurementObject::clone();
}

Vector3f AngleMeasurementObject::getWorldPoint() const
{
    Vector3f ret = getLocalPoint();
    if ( parent() )
        ret = parent()->worldXf()( ret );
    return ret;
}

Vector3f AngleMeasurementObject::getLocalPoint() const
{
    return xf().b;
}

Vector3f AngleMeasurementObject::getWorldRay( bool second ) const
{
    Vector3f ret = getLocalRay( second );
    if ( parent() )
        ret = parent()->worldXf().A * ret;
    return ret;
}

Vector3f AngleMeasurementObject::getLocalRay( bool second ) const
{
    return xf().A.col( second ? 1 : 0 );
}

void AngleMeasurementObject::setLocalPoint( const MR::Vector3f& point )
{
    auto curXf = xf();
    curXf.b = point;
    setXf( curXf );
}

void AngleMeasurementObject::setLocalRays( const MR::Vector3f& a, const MR::Vector3f& b )
{
    auto curXf = xf();
    Vector3f c = cross( a, b );
    if ( c == Vector3f{} )
        c = cross( a, a.furthestBasisVector() ); // Some random fallback direction.
    c = c.normalized();
    curXf.A = Matrix3f( a, b, c ).transposed();
    setXf( curXf );
}

bool AngleMeasurementObject::getIsConical() const
{
    return isConical_;
}

void AngleMeasurementObject::setIsConical( bool value )
{
    isConical_ = value;
}

bool AngleMeasurementObject::getShouldVisualizeRay( bool second ) const
{
    return shouldVisualizeRay_[second];
}

void AngleMeasurementObject::setShouldVisualizeRay( bool second, bool enable )
{
    shouldVisualizeRay_[second] = enable;
}

float AngleMeasurementObject::computeAngle() const
{
    if ( !cachedValue_ )
        cachedValue_ = angle( getWorldRay( false ), getWorldRay( true ) );
    return *cachedValue_;
}

std::vector<std::string> AngleMeasurementObject::getInfoLines() const
{
    auto ret = MeasurementObject::getInfoLines();
    ret.push_back( fmt::format( "angle value: {:.1f}\xC2\xB0", computeAngle() ) ); // U+00B0 DEGREE SIGN
    return ret;
}

void AngleMeasurementObject::swapBase_( Object& other )
{
    if ( auto ptr = other.asType<AngleMeasurementObject>() )
        std::swap( *this, *ptr );
    else
        assert( false );
}

void AngleMeasurementObject::serializeFields_( Json::Value& root ) const
{
    MeasurementObject::serializeFields_( root );
    root["Type"].append( TypeName() );

    root["IsConical"] = isConical_;

    root["ShouldVisualizeRayA"] = shouldVisualizeRay_[0];
    root["ShouldVisualizeRayB"] = shouldVisualizeRay_[1];
}

void AngleMeasurementObject::deserializeFields_( const Json::Value& root )
{
    MeasurementObject::deserializeFields_( root );

    if ( const auto& json = root["IsConical"]; json.isBool() )
        isConical_ = json.asBool();

    if ( const auto& json = root["ShouldVisualizeRayA"]; json.isBool() )
        shouldVisualizeRay_[0] = json.asBool();
    if ( const auto& json = root["ShouldVisualizeRayB"]; json.isBool() )
        shouldVisualizeRay_[1] = json.asBool();
}

void AngleMeasurementObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

void AngleMeasurementObject::propagateWorldXfChangedSignal_()
{
    MeasurementObject::propagateWorldXfChangedSignal_();
    cachedValue_ = {};
}

} // namespace MR
