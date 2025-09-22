#include "MRRenderMeasurementObjects.h"
#include "MRUnits.h"

#include "MRMesh/MRFeatureObject.h"
#include "MRMesh/MRVisualObject.h"
#include "MRPch/MRFmt.h"

namespace MR
{

// Returns the UI color for the measurement `object`.
static Color getMeasurementColor( const VisualObject& object, ViewportId viewport )
{
    // If our parent is a feature and we're not selected, copy its color.
    if ( !object.isSelected() )
    {
        if ( auto feature = dynamic_cast<const FeatureObject*>( object.parent() ) )
            return feature->getFrontColor( feature->isSelected(), viewport );
    }

    // Otherwise use our own color.
    return object.getFrontColor( object.isSelected(), viewport );
}

MR_REGISTER_RENDER_OBJECT_IMPL( DistanceMeasurementObject, RenderDistanceObject )
RenderDistanceObject::RenderDistanceObject( const VisualObject& object )
    : RenderDimensionObject( object ), object_( &dynamic_cast<const DistanceMeasurementObject&>( object ) )
{}

void RenderDistanceObject::renderUi( const UiRenderParams& params )
{
    Vector3f pointA = object_->getWorldPoint();
    Vector3f pointB = pointA + object_->getWorldDelta();
    auto ref = object_->getComparisonReferenceValue( 0 );
    auto tol = object_->getComparisonTolerence( 0 );
    task_ = RenderDimensions::LengthTask( params, {}, getMeasurementColor( *object_, params.viewportId ), {
        .common = {
            .objectToSelect = object_,
            .objectName = object_->name(),
        },
        .points = { pointA, pointB },
        .drawAsNegative = object_->isNegative(),
        .onlyOneAxis =
            object_->getDistanceMode() == DistanceMeasurementObject::DistanceMode::xAbsolute ? std::optional( 0 ) :
            object_->getDistanceMode() == DistanceMeasurementObject::DistanceMode::yAbsolute ? std::optional( 1 ) :
            object_->getDistanceMode() == DistanceMeasurementObject::DistanceMode::zAbsolute ? std::optional( 2 ) : std::nullopt,
        .referenceValue =
            ref.isSet ? std::optional( std::get<float>( ref.var ) ) : std::nullopt,
        .tolerance =
            tol ? std::optional( RenderDimensions::LengthParams::Tolerance{ .positive = tol->positive, .negative = tol->negative } ) : std::nullopt,
    } );
    params.tasks->push_back( { std::shared_ptr<void>{}, &task_ } ); // A non-owning shared pointer.
}

MR_REGISTER_RENDER_OBJECT_IMPL( RadiusMeasurementObject, RenderRadiusObject )
RenderRadiusObject::RenderRadiusObject( const VisualObject& object )
    : RenderDimensionObject( object ), object_( &dynamic_cast<const RadiusMeasurementObject&>( object ) )
{}

void RenderRadiusObject::renderUi( const UiRenderParams& params )
{
    task_ = RenderDimensions::RadiusTask( params, {}, getMeasurementColor( *object_, params.viewportId ), {
        .common = {
            .objectToSelect = object_,
            .objectName = object_->name(),
        },
        .center = object_->getWorldCenter(),
        .radiusAsVector = object_->getWorldRadiusAsVector(),
        .normal = object_->getWorldNormal(),
        .drawAsDiameter = object_->getDrawAsDiameter(),
        .isSpherical = object_->getIsSpherical(),
        .visualLengthMultiplier = object_->getVisualLengthMultiplier(),
    } );
    params.tasks->push_back( { std::shared_ptr<void>{}, &task_ } ); // A non-owning shared pointer.
}

MR_REGISTER_RENDER_OBJECT_IMPL( AngleMeasurementObject, RenderAngleObject )
RenderAngleObject::RenderAngleObject( const VisualObject& object )
    : RenderDimensionObject( object ), object_( &dynamic_cast<const AngleMeasurementObject&>( object ) )
{}

void RenderAngleObject::renderUi( const UiRenderParams& params )
{
    task_ = RenderDimensions::AngleTask( params, {}, getMeasurementColor( *object_, params.viewportId ), {
        .common = {
            .objectToSelect = object_,
            .objectName = object_->name(),
        },
        .center = object_->getWorldPoint(),
        .rays = {
            object_->getWorldRay( false ),
            object_->getWorldRay( true ),
        },
        .isConical = object_->getIsConical(),
        .shouldVisualizeRay = {
            object_->getShouldVisualizeRay( false ),
            object_->getShouldVisualizeRay( true ),
        },
    } );
    params.tasks->push_back( { std::shared_ptr<void>{}, &task_ } ); // A non-owning shared pointer.
}

MR_REGISTER_RENDER_OBJECT_IMPL( PointMeasurementObject, RenderPointMeasurementObject )

RenderPointMeasurementObject::RenderPointMeasurementObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    nameUiScreenOffset = Vector2f( 0.1f, 0.1f );
    nameUiPointIsRelativeToBoundingBoxCenter = false;
}

ImGuiMeasurementIndicators::Text RenderPointMeasurementObject::getObjectNameText( const VisualObject& object, ViewportId ) const
{
    ImGuiMeasurementIndicators::Text result;

    result.addText( object.name() );

    if ( const auto* pointObj = dynamic_cast<const PointMeasurementObject*>( &object ) )
    {
        const auto refNormalValue = pointObj->getComparisonReferenceValue( 1 );
        const auto* refNormal = std::get_if<Vector3f>( &refNormalValue.var );
        const auto printValue = [&] ( const Vector3f& p )
        {
            static const auto toString = [] ( float v )
            {
                return valueToString<NoUnit>( v, {
                    .stripTrailingZeroes = false,
                } );
            };
            if ( refNormalValue.isSet )
            {
                assert( refNormal );
                if ( *refNormal == Vector3f::plusX() || *refNormal == Vector3f::minusX() )
                    return fmt::format( "X {}", toString( p.x ) );
                if ( *refNormal == Vector3f::plusY() || *refNormal == Vector3f::minusY() )
                    return fmt::format( "Y {}", toString( p.y ) );
                if ( *refNormal == Vector3f::plusZ() || *refNormal == Vector3f::minusZ() )
                    return fmt::format( "Z {}", toString( p.z ) );
            }
            return fmt::format( "{} {} {}", toString( p.x ), toString( p.y ), toString( p.z ) );
        };
        const auto printTolerance = [&]
        {
            // `dir == 0` - symmetric, `dir > 0` - positive, `dir < 0` - negative.
            static const auto toString = [] ( float value, int dir )
            {
                return valueToString<LengthUnit>( value, {
                    .unitSuffix = false,
                    .style = NumberStyle::normal,
                    .plusSign = dir != 0,
                    .zeroMode = dir >= 0 ? ZeroMode::alwaysPositive : ZeroMode::alwaysNegative,
                    .stripTrailingZeroes = true,
                } );
            };
            if ( const auto tol = pointObj->getComparisonTolerence( 0 ) )
            {
                if ( tol->positive == -tol->negative )
                    return fmt::format( " \xC2\xB1{}", toString( tol->positive, 0 ) ); // U+00B1 PLUS-MINUS SIGN
                else
                    return fmt::format( " {}/{}", toString( tol->positive, 1 ), toString( tol->negative, -1 ) );
            }
            return std::string{};
        };

        const auto p = pointObj->getPoint();
        if ( const auto refPosValue = pointObj->getComparisonReferenceValue( 0 ); refPosValue.isSet )
        {
            const auto* refPos = std::get_if<Vector3f>( &refPosValue.var );
            assert( refPos );
            result.addLine();
            result.addText( fmt::format( "Measured: {}", printValue( p ) ) );
            result.addLine();
            result.addText( fmt::format( "Nominal:  {}{}", printValue( *refPos ), printTolerance() ) );
        }
        else
        {
            result.addLine();
            result.addText( printValue( p ) );
        }
    }

    return result;
}

}
