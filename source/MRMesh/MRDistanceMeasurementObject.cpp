#include "MRDistanceMeasurementObject.h"

#include "MRMesh/MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRFmt.h"

#include <cassert>

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

bool DistanceMeasurementObject::isNegative() const
{
    return isNegative_;
}

void DistanceMeasurementObject::setIsNegative( bool value )
{
    if ( isNegative_ != value )
    {
        isNegative_ = value;
        cachedValue_ = {};
    }
}

DistanceMeasurementObject::DistanceMode DistanceMeasurementObject::getDistanceMode() const
{
    return perCoordDeltas_;
}

void DistanceMeasurementObject::setDistanceMode( DistanceMode mode )
{
    if ( mode != perCoordDeltas_ )
    {
        perCoordDeltas_ = mode;
        cachedValue_ = {};
    }
}

float DistanceMeasurementObject::computeDistance() const
{
    if ( !cachedValue_ )
    {
        int axis = -1;
        switch ( perCoordDeltas_ )
        {
        case DistanceMode::xAbsolute:
            axis = 0;
            break;
        case DistanceMode::yAbsolute:
            axis = 1;
            break;
        case DistanceMode::zAbsolute:
            axis = 2;
            break;
        default:
            // Nothing.
            break;
        }
        cachedValue_ = ( axis == -1 ? getWorldDelta().length() : std::abs( getWorldDelta()[axis] ) ) * ( isNegative() ? -1.f : 1.f );
    }
    return *cachedValue_;
}

std::vector<std::string> DistanceMeasurementObject::getInfoLines() const
{
    auto ret = MeasurementObject::getInfoLines();
    ret.push_back( fmt::format( "distance value: {:.3f}", computeDistance() ) );
    return ret;
}

std::size_t DistanceMeasurementObject::numComparableProperties() const
{
    return 1;
}

std::string_view DistanceMeasurementObject::getComparablePropertyName( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );

    switch ( perCoordDeltas_ )
    {
    case DistanceMode::xAbsolute:
        return "X Distance";
    case DistanceMode::yAbsolute:
        return "Y Distance";
    case DistanceMode::zAbsolute:
        return "Z Distance";
    default:
        return "Distance";
    }
}

std::optional<DistanceMeasurementObject::ComparableProperty> DistanceMeasurementObject::computeComparableProperty( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    return ComparableProperty{
        .value = computeDistance(),
        .referenceValue = referenceValue_,
    };
}

std::optional<DistanceMeasurementObject::ComparisonTolerance> DistanceMeasurementObject::getComparisonTolerence( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    return tolerance_;
}

void DistanceMeasurementObject::setComparisonTolerance( std::size_t i, std::optional<ComparisonTolerance> newTolerance )
{
    (void)i;
    assert( i == 0 );
    tolerance_ = newTolerance;
}

std::string_view DistanceMeasurementObject::getComparisonReferenceValueName( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    return "Nominal";
}

DistanceMeasurementObject::ComparisonReferenceValue DistanceMeasurementObject::getComparisonReferenceValue( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    return { .isSet = bool( referenceValue_ ), .var = referenceValue_.value_or( 0.f ) };
}

void DistanceMeasurementObject::setComparisonReferenceValue( std::size_t i, std::optional<ComparisonReferenceValue::Var> value )
{
    (void)i;
    assert( i == 0 );
    if ( value )
    {
        auto ptr = std::get_if<float>( &*value );
        assert( ptr );
        if ( ptr )
            referenceValue_ = *ptr;
    }
    else
    {
        referenceValue_.reset();
    }
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

    root["DrawAsNegative"] = isNegative_;

    if ( tolerance_ )
    {
        root["TolerancePositive"] = tolerance_->positive;
        root["ToleranceNegative"] = tolerance_->negative;
    }
    else
    {
        root["TolerancePositive"] = Json::nullValue;
        root["ToleranceNegative"] = Json::nullValue;
    }

    if ( referenceValue_ )
        root["ReferenceValue"] = *referenceValue_;
    else
        root["ReferenceValue"] = Json::nullValue;
}

void DistanceMeasurementObject::deserializeFields_( const Json::Value& root )
{
    MeasurementObject::deserializeFields_( root );

    if ( const auto& json = root["DrawAsNegative"]; json.isBool() )
        isNegative_ = json.asBool();

    if ( const auto& json = root["DistanceMode"]; json.isInt() )
        perCoordDeltas_ = DistanceMode( json.asInt() );

    { // Tolerance.
        const auto& jsonPos = root["TolerancePositive"];
        const auto& jsonNeg = root["ToleranceNegative"];
        if ( jsonPos.isDouble() && jsonNeg.isDouble() )
        {
            tolerance_.emplace();
            tolerance_->positive = jsonPos.asFloat();
            tolerance_->negative = jsonNeg.asFloat();
        }
    }

    if ( const auto& json = root["ReferenceValue"]; json.isDouble() )
        referenceValue_ = json.asFloat();
}

void DistanceMeasurementObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

void DistanceMeasurementObject::onWorldXfChanged_()
{
    MeasurementObject::onWorldXfChanged_();
    cachedValue_ = {};
}

} // namespace MR
