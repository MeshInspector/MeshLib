#include "MRPointObject.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"
#include "MRVector3.h"

#include <cassert>

namespace MR
{

MR_ADD_CLASS_FACTORY( PointObject )

PointObject::PointObject()
    : FeatureObject( 0 )
{}

PointObject::PointObject( const std::vector<Vector3f>& pointsToApprox )
    : PointObject()
{
    Vector3d center;
    for ( auto& p : pointsToApprox )
        center += Vector3d( p );
    setPoint( Vector3f( center / double( pointsToApprox.size() ) ) );
}

std::shared_ptr<MR::Object> PointObject::clone() const
{
    return std::make_shared<PointObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<MR::Object> PointObject::shallowClone() const
{
    return std::make_shared<PointObject>( ProtectedStruct{}, *this );
}

Vector3f PointObject::getPoint( ViewportId id /*= {}*/ ) const
{
    return xf( id ).b;
}

void PointObject::setPoint( const Vector3f& point, ViewportId id /*= {}*/ )
{
    setXf( AffineXf3f::translation( point ), id );
}

std::vector<FeatureObjectSharedProperty>& PointObject::getAllSharedProperties() const
{
    static std::vector<FeatureObjectSharedProperty> ret = {
       {"Point", FeaturePropertyKind::position, &PointObject::getPoint, &PointObject::setPoint}
    };
    return ret;
}

FeatureObjectProjectPointResult PointObject::projectPoint( const Vector3f& /*point*/, ViewportId id /*= {}*/ ) const
{
    return { getPoint( id ), std::nullopt };
}

std::size_t PointObject::numComparableProperties() const
{
    return 1;
}

std::string_view PointObject::getComparablePropertyName( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    return "Deviation";
}

std::optional<PointObject::ComparableProperty> PointObject::computeComparableProperty( std::size_t i ) const
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

std::optional<PointObject::ComparisonTolerance> PointObject::getComparisonTolerence( std::size_t i ) const
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

void PointObject::setComparisonTolerance( std::size_t i, std::optional<ComparisonTolerance> newTolerance )
{
    (void)i;
    assert( i == 0 );
    tolerance_ = newTolerance;
}

bool PointObject::comparisonToleranceIsAlwaysOnlyPositive( std::size_t i ) const
{
    (void)i;
    assert( i == 0 );
    return !bool( referenceNormal_ ); // If we don't have a reference normal, we calculate the euclidean distance, which can't be negative.
}

std::size_t PointObject::numComparisonReferenceValues() const
{
    return 2;
}

std::string_view PointObject::getComparisonReferenceValueName( std::size_t i ) const
{
    assert( i < 2 );
    return std::array{ "Nominal pos", "Direction" }[i];
}

PointObject::ComparisonReferenceValue PointObject::getComparisonReferenceValue( std::size_t i ) const
{
    assert( i < 2 );
    const auto& target = i == 1 ? referenceNormal_ : referencePos_;
    return { .isSet = bool( target ), .var = target ? *target : Vector3f{} };
}

void PointObject::setComparisonReferenceValue( std::size_t i, std::optional<ComparisonReferenceValue::Var> value )
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

void PointObject::swapBase_( Object& other )
{
    if ( auto pointObject = other.asType<PointObject>() )
        std::swap( *this, *pointObject );
    else
        assert( false );
}

void PointObject::serializeFields_( Json::Value& root ) const
{
    FeatureObject::serializeFields_( root );
    root["Type"].append( PointObject::TypeName() );
}

void PointObject::setupRenderObject_() const
{
    if ( !renderObj_ )
        renderObj_ = createRenderObject<decltype( *this )>( *this );
}

}
