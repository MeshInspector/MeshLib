#include "MRRenderMeasurementObjects.h"
#include "MRMesh/MRUnits.h"
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

MR_REGISTER_RENDER_OBJECT_IMPL( PointMeasurementObject, RenderPointObject )
RenderPointObject::RenderPointObject( const VisualObject& object )
    : RenderObjectCombinator( object ), object_( &dynamic_cast<const PointMeasurementObject&>( object ) )
{}

void RenderPointObject::renderUi( const UiRenderParams& params )
{
    auto refPoint = object_->getComparisonReferenceValue( 0 );
    auto refNormal = object_->getComparisonReferenceValue( 1 );
    auto tol = object_->getComparisonTolerence( 0 );
    task_ = RenderDimensions::PointTask( params, {}, getMeasurementColor( *object_, params.viewportId ), RenderDimensions::PointParams{
        .common = {
            .objectToSelect = object_,
            .objectName = object_->name(),
        },
        .point = object_->getWorldPoint( params.viewportId ),
        .align = ImVec2( 1, 1 ),
        .referencePoint = refPoint.isSet ? std::optional( std::get<Vector3f>( refPoint.var ) ) : std::nullopt,
        .referenceNormal = refNormal.isSet ? std::get<Vector3f>( refNormal.var ) : Vector3f{},
        .tolerance = tol ? std::optional( RenderDimensions::Tolerance{ .positive = tol->positive, .negative = tol->negative } ) : std::nullopt,
        .capIsVisible = object_->getVisualizeProperty( PointMeasurementVisualizePropertyType::CapVisibility, params.viewportId ),
    } );
    params.tasks->push_back( { std::shared_ptr<void>{}, &task_ } ); // A non-owning shared pointer.
}

MR_REGISTER_RENDER_OBJECT_IMPL( DistanceMeasurementObject, RenderDistanceObject )
RenderDistanceObject::RenderDistanceObject( const VisualObject& object )
    : RenderObjectCombinator( object ), object_( &dynamic_cast<const DistanceMeasurementObject&>( object ) )
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
            tol ? std::optional( RenderDimensions::Tolerance{ .positive = tol->positive, .negative = tol->negative } ) : std::nullopt,
    } );
    params.tasks->push_back( { std::shared_ptr<void>{}, &task_ } ); // A non-owning shared pointer.
}

MR_REGISTER_RENDER_OBJECT_IMPL( RadiusMeasurementObject, RenderRadiusObject )
RenderRadiusObject::RenderRadiusObject( const VisualObject& object )
    : RenderObjectCombinator( object ), object_( &dynamic_cast<const RadiusMeasurementObject&>( object ) )
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
    : RenderObjectCombinator( object ), object_( &dynamic_cast<const AngleMeasurementObject&>( object ) )
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
