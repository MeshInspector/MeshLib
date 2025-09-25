#include "MRPointMeasurementObject.h"

#include "MRObjectFactory.h"
#include "MRSerializer.h"

#include "MRPch/MRJson.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( PointMeasurementObject )

std::shared_ptr<Object> PointMeasurementObject::clone() const
{
    return std::make_shared<PointMeasurementObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<Object> PointMeasurementObject::shallowClone() const
{
    return std::make_shared<PointMeasurementObject>( ProtectedStruct{}, *this );
}

bool PointMeasurementObject::supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const
{
    return VisualObject::supportsVisualizeProperty( type ) || type.tryGet<PointMeasurementVisualizePropertyType>().has_value();
}

AllVisualizeProperties PointMeasurementObject::getAllVisualizeProperties() const
{
    AllVisualizeProperties ret = VisualObject::getAllVisualizeProperties();
    getAllVisualizePropertiesForEnum<PointMeasurementVisualizePropertyType>( ret );
    return ret;
}

const ViewportMask& PointMeasurementObject::getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const
{
    if ( auto value = type.tryGet<PointMeasurementVisualizePropertyType>() )
    {
        switch ( *value )
        {
        case PointMeasurementVisualizePropertyType::CapVisibility:
            return capVisibility_;
        case PointMeasurementVisualizePropertyType::_count:
            MR_UNREACHABLE_NO_RETURN
        }
        MR_UNREACHABLE_NO_RETURN
        return visibilityMask_;
    }
    else
    {
        return VisualObject::getVisualizePropertyMask( type );
    }
}

Vector3f PointMeasurementObject::getLocalPoint( ViewportId id ) const
{
    return xf( id ).b;
}

Vector3f PointMeasurementObject::getWorldPoint( ViewportId id ) const
{
    Vector3f ret = getLocalPoint( id );
    if ( parent() )
        ret = parent()->worldXf( id )( ret );
    return ret;
}

void PointMeasurementObject::setLocalPoint( const Vector3f& point, ViewportId id )
{
    setXf( AffineXf3f::translation( point ), id );
}

void PointMeasurementObject::setWorldPoint( const Vector3f& point, ViewportId id )
{
    setWorldXf( AffineXf3f::translation( point ), id );
}

std::size_t PointMeasurementObject::numComparableProperties() const
{
    return 1;
}

std::string_view PointMeasurementObject::getComparablePropertyName( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    return "Deviation";
}

std::optional<PointMeasurementObject::ComparableProperty> PointMeasurementObject::computeComparableProperty( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );

    if ( !referencePos_ )
        return {}; // Can't compute the distance without knowing the reference point.

    // World or local position? I assume we want the world one...
    Vector3f thisPos = worldXf().b;

    // Do we use the normal?
    if ( referenceNormal_ )
    {
        // With normal:

        if ( *referenceNormal_ == Vector3f{} )
            return {}; // Zero normal invalid.

        return ComparableProperty{
            .value = dot( thisPos - *referencePos_, *referenceNormal_ ) / referenceNormal_->length(),
            .referenceValue = 0.f, // Always zero for points for now.
        };
    }
    else
    {
        // Without normal:

        return ComparableProperty{
            .value = ( thisPos - *referencePos_ ).length(),
            .referenceValue = 0.f, // Always zero for points for now.
        };
    }
}

std::optional<PointMeasurementObject::ComparisonTolerance> PointMeasurementObject::getComparisonTolerence( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    if ( comparisonToleranceIsAlwaysOnlyPositive( i ) )
    {
        assert( !tolerance_ || tolerance_->negative == 0 );
        return tolerance_ ? std::optional( ComparisonTolerance{ .positive = tolerance_->positive } ) : std::nullopt;
    }
    else
    {
        return tolerance_;
    }
}

void PointMeasurementObject::setComparisonTolerance( std::size_t i, std::optional<ComparisonTolerance> newTolerance )
{
    (void)i;
    assert( i == 0 );
    tolerance_ = newTolerance;
}

bool PointMeasurementObject::comparisonToleranceIsAlwaysOnlyPositive( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    return !bool( referenceNormal_ ); // If we don't have a reference normal, we calculate the Euclidean distance, which can't be negative.
}

std::size_t PointMeasurementObject::numComparisonReferenceValues() const
{
    return 2;
}

std::string_view PointMeasurementObject::getComparisonReferenceValueName( std::size_t i ) const
{
    assert( i < 2 );
    return std::array{ "Nominal pos", "Direction" }[i];
}

PointMeasurementObject::ComparisonReferenceValue PointMeasurementObject::getComparisonReferenceValue( std::size_t i ) const
{
    assert( i < 2 );
    const auto& target = i == 1 ? referenceNormal_ : referencePos_;
    return { .isSet = bool( target ), .var = target ? *target : Vector3f{} };
}

void PointMeasurementObject::setComparisonReferenceValue( std::size_t i, std::optional<ComparisonReferenceValue::Var> value )
{
    assert( i < 2 );
    auto& target = ( i == 1 ? referenceNormal_ : referencePos_ );
    if ( value )
    {
        // When adding the normal, make the tolerance symmetric again (before that, the negative part should've been zeroed). See also `comparisonToleranceIsAlwaysOnlyPositive()`.
        if ( &target == &referenceNormal_ && !target && tolerance_ )
            tolerance_->negative = -tolerance_->positive;

        auto ptr = std::get_if<Vector3f>( &*value );
        assert( ptr );
        target = *ptr;
    }
    else
    {
        target = {};

        // When removing the normal, also zero the negative tolerance. See also `comparisonToleranceIsAlwaysOnlyPositive()`.
        if ( &target == &referenceNormal_ && tolerance_ )
            tolerance_->negative = 0;
    }
}

void PointMeasurementObject::swapBase_( Object& other )
{
    if ( auto pointObject = other.asType<PointMeasurementObject>() )
        std::swap( *this, *pointObject );
    else
        assert( false );
}

void PointMeasurementObject::serializeFields_( Json::Value& root ) const
{
    MeasurementObject::serializeFields_( root );
    root["Type"].append( TypeName() );

    if ( referencePos_ )
        serializeToJson( *referencePos_, root["ReferencePos"] );
    else
        root["ReferencePos"] = Json::nullValue;

    if ( referenceNormal_ )
        serializeToJson( *referenceNormal_, root["ReferenceNormal"] );
    else
        root["ReferenceNormal"] = Json::nullValue;

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
}

void PointMeasurementObject::deserializeFields_( const Json::Value& root )
{
    MeasurementObject::deserializeFields_( root );

    if ( const auto& json = root["ReferencePos"]; json.isObject() )
    {
        referencePos_.emplace();
        deserializeFromJson( json, *referencePos_ );
    }

    if ( const auto& json = root["ReferenceNormal"]; json.isObject() )
    {
        referenceNormal_.emplace();
        deserializeFromJson( json, *referenceNormal_ );
    }

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
}

void PointMeasurementObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

} // namespace MR
