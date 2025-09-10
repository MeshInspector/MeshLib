#include "MRRenderMeasurementObjects.h"

#include "MRMesh/MRFeatureObject.h"
#include "MRMesh/MRVisualObject.h"

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

}
