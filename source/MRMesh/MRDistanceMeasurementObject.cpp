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
    perCoordDeltas_ = mode;
}

float DistanceMeasurementObject::computeDistance() const
{
    if ( !cachedValue_ )
        cachedValue_ = getWorldDelta().length() * ( isNegative() ? -1.f : 1.f );
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
    return "Distance";
}

std::optional<float> DistanceMeasurementObject::compareProperty( const Object& other, std::size_t i ) const
{
    (void)i;
    assert( i == 0 );

    auto otherDistance = dynamic_cast<const DistanceMeasurementObject*>( &other );
    assert( otherDistance );
    if ( !otherDistance )
        return {};

    return computeDistance() - otherDistance->computeDistance();
}

bool DistanceMeasurementObject::hasComparisonTolerances() const
{
    return bool( tolerance_ );
}

DistanceMeasurementObject::ComparisonTolerance DistanceMeasurementObject::getComparisonTolerences( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    assert( bool( tolerance_ ) );
    return tolerance_ ? *tolerance_ : ComparisonTolerance{};
}

void DistanceMeasurementObject::setComparisonTolerance( std::size_t i, const ComparisonTolerance& newTolerance )
{
    (void)i;
    assert( i == 0 );
    tolerance_ = newTolerance;
}

void DistanceMeasurementObject::resetComparisonTolerances()
{
    tolerance_.reset();
}

bool DistanceMeasurementObject::hasComparisonReferenceValues() const
{
    return bool( referenceValue_ );
}

float DistanceMeasurementObject::getComparisonReferenceValue( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    assert( referenceValue_ );
    return referenceValue_.value_or( 0.f );
}

void DistanceMeasurementObject::setComparisonReferenceValue( std::size_t i, float value )
{
    (void)i;
    assert( i == 0 );
    referenceValue_ = value;
}

void DistanceMeasurementObject::resetComparisonReferenceValues()
{
    return referenceValue_.reset();
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
