#include "MRRenderFeatureObjects.h"
#include "MRVisualSubfeatures.h"

#include "MRMesh/MRArrow.h"
#include "MRMesh/MRCircleObject.h"
#include "MRMesh/MRConeObject.h"
#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRFeatures.h"
#include "MRMesh/MRLineObject.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRMeshNormals.h"
#include "MRMesh/MRPlaneObject.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointObject.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRSphereObject.h"
#include "MRMesh/MRSubfeatures.h"
#include "MRPch/MRFmt.h"

namespace MR::RenderFeatures
{

static constexpr int numCircleSegments = 128;
static constexpr int sphereDetailLevel = 2048;

// Separator between object name and extra information.
static constexpr std::string_view nameExtrasSeparator = "   |   ";

// Extracts subfeatures from `sourceObject` and writes them out `outputPoints` and `outputLines`.
// `sourceObject` must point to a temporary object of the desired type, with identity xf.
static void addSubfeatures( const VisualObject& sourceObject, ObjectLines* outputLines, ObjectPoints* outputPoints )
{
    // Actually add the subfeatures:

    auto parentFeature = Features::primitiveFromObjectWithWorldXf( sourceObject );
    assert( parentFeature && "Can't convert this object to a feature" );
    if ( !parentFeature )
        return;

    Features::forEachVisualSubfeature( *parentFeature, [&]( const Features::SubfeatureInfo& params ) -> void
    {
        // It's a bit jank to utilize `primitiveToObject()` just to switch over the subfeature types, but it's very convenient.

        if ( params.isInfinite )
            return;

        const auto subFeature = params.create( *parentFeature );

        constexpr float infiniteExtent = 10; // Whatever, we shouldn't receive any infinite features anyway.
        auto subObject = Features::primitiveToObject( subFeature, infiniteExtent );
        if ( !subObject )
        {
            assert( false && "Unknown subfeature type." );
            return;
        }

        if ( auto point = dynamic_cast<PointObject *>( subObject.get() ) )
        {
            outputPoints->varPointCloud()->addPoint( point->getPoint() );
            return;
        }
        if ( auto line = dynamic_cast<LineObject *>( subObject.get() ) )
        {
            outputLines->varPolyline()->addFromPoints( std::array{ line->getPointA(), line->getPointB() }.data(), 2, false );
            return;
        }
        if ( auto circle = dynamic_cast<CircleObject *>( subObject.get() ) )
        {
            std::array<Vector3f, numCircleSegments> points;
            for ( int i = 0; i < numCircleSegments; ++i )
            {
                float angle = i * 2 * PI_F / numCircleSegments;
                points[i].x = cosf( angle );
                points[i].y = sinf( angle );
                points[i] = circle->xf()( points[i] );
            }
            outputLines->varPolyline()->addFromPoints( points.data(), numCircleSegments, true );
            return;
        }

        assert( false && "The subfeature uses an unknown object type." );
    } );
}


RenderPlaneNormalComponent::RenderPlaneNormalComponent( const VisualObject& object )
    : RenderFeatureMeshComponent( object )
{
    static const auto mesh = []{
        return std::make_shared<Mesh>( makeArrow( Vector3f( 0, 0, 0 ), Vector3f( 0, 0, 1 ), 0.035f, 0.07f, 0.14f, numCircleSegments ) );
    }();
    subobject.setMesh( mesh );
    subobject.setFlatShading( true );
}

bool RenderPlaneNormalComponent::render( const ModelRenderParams& params )
{
    Matrix3f planeScaleMat = dynamic_cast<const FeatureObject *>( subobject.target_ )->getScaleShearMatrix();
    float normalScale = std::min( planeScaleMat.x.x, planeScaleMat.y.y ) * ( 2 / 3.f );

    Matrix4f newModelMatrix =
        subobject.target_->worldXf( params.viewportId ) *
        AffineXf3f::translation( Vector3f( -1, -1, 0 ) ) *
        AffineXf3f::linear( Matrix3f::scale( Vector3f( normalScale / planeScaleMat.x.x, normalScale / planeScaleMat.y.y, normalScale / planeScaleMat.z.z ) ) );
    ModelRenderParams newParams = {
        {
            static_cast<const BaseRenderParams &>( params ),
            newModelMatrix,
            params.clipPlane,
            params.depthFunction,
        },
        params.normMatrixPtr,
        params.lightPos,
        params.allowAlphaSort,
        params.passMask,
    };

    return RenderFeatureMeshComponent::render( newParams );
}

void RenderPlaneNormalComponent::renderPicker( const ModelBaseRenderParams& params, unsigned geomId )
{
    // No picking for the normal!
    (void)params;
    (void)geomId;
}


MR_REGISTER_RENDER_OBJECT_IMPL( PointObject, RenderPointFeatureObject )
RenderPointFeatureObject::RenderPointFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    static const auto pointCloud = []{
        auto ret = std::make_shared<PointCloud>();
        ret->addPoint( Vector3f{} );
        return ret;
    }();
    subobject.setPointCloud( pointCloud );

    nameUiPointIsRelativeToBoundingBoxCenter = false;
    nameUiScreenOffset = Vector2f( 0, 0.1f );
}

std::string RenderPointFeatureObject::getObjectNameString( const VisualObject& object, ViewportId viewportId ) const
{
    if ( object.getVisualizeProperty( FeatureVisualizePropertyType::DetailsOnNameTag, viewportId ) )
    {
        Vector3f point = object.xf().b;
        if ( object.parent() )
            point = object.parent()->worldXf()( point );
        constexpr int precision = 2;
        return fmt::format( "{}{}{:.{}f}, {:.{}f}, {:.{}f}", RenderObjectCombinator::getObjectNameString( object, viewportId ), nameExtrasSeparator, point.x, precision, point.y, precision, point.z, precision );
    }
    else
    {
        return RenderObjectCombinator::getObjectNameString( object, viewportId );
    }
}

MR_REGISTER_RENDER_OBJECT_IMPL( LineObject, RenderLineFeatureObject )
RenderLineFeatureObject::RenderLineFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    static const auto polyline = []{
        auto ret = std::make_shared<Polyline3>();
        std::array points = { Vector3f( -1, 0, 0 ), Vector3f( 1, 0, 0 ) };
        ret->addFromPoints( points.data(), points.size() );
        return ret;
    }();
    subobject.setPolyline( polyline );

    nameUiPointIsRelativeToBoundingBoxCenter = false;
    nameUiLocalOffset = Vector3f( 0.01f, 0, 0 );
    nameUiRotateLocalOffset90Degrees = true;
}

std::string RenderLineFeatureObject::getObjectNameString( const VisualObject& object, ViewportId viewportId ) const
{
    if ( object.getVisualizeProperty( FeatureVisualizePropertyType::DetailsOnNameTag, viewportId ) )
    {
        Vector3f dir = object.xf().A.col( 0 );
        if ( object.parent() )
            dir = object.parent()->worldXf().A * dir;
        dir = dir.normalized();
        constexpr int precision = 2;

        return fmt::format( "{}{}dir {:.{}f}, {:.{}f}, {:.{}f}", RenderObjectCombinator::getObjectNameString( object, viewportId ), nameExtrasSeparator, dir.x, precision, dir.y, precision, dir.z, precision );
    }
    else
    {
        return RenderObjectCombinator::getObjectNameString( object, viewportId );
    }
}

MR_REGISTER_RENDER_OBJECT_IMPL( CircleObject, RenderCircleFeatureObject )
RenderCircleFeatureObject::RenderCircleFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object ), object_( &object )
{
    // Main visualization.
    static const auto polyline = []{
        auto ret = std::make_shared<Polyline3>();
        std::array<Vector3f, numCircleSegments> points;
        for ( int i = 0; i < numCircleSegments; ++i )
        {
            float angle = i * 2 * PI_F / numCircleSegments;
            points[i].x = cosf( angle );
            points[i].y = sinf( angle );
        }
        ret->addFromPoints( points.data(), numCircleSegments, true );
        return ret;
    }();
    getLines().setPolyline( polyline );

    // Subfeatures.
    getPoints().setPointCloud( std::make_shared<PointCloud>() );
    addSubfeatures( CircleObject{}, &getLines(), &getPoints() );

    // More or less an arbitrary direction. Just something that's not +X to avoid overlaps with other stuff.
    Vector3f nameTagDir = Vector3f( -1, -1, 0 ).normalized();
    nameUiPoint = nameTagDir;
    nameUiPointIsRelativeToBoundingBoxCenter = false;
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
}

void RenderCircleFeatureObject::renderUi( const UiRenderParams& params )
{
    RenderObjectCombinator::renderUi( params );

    if ( object_->getVisualizeProperty( DimensionsVisualizePropertyType::diameter, params.viewportId ) )
    {
        radiusTask_ = RenderDimensions::RadiusTask( params, object_->worldXf(), object_->getFrontColor( object_->isSelected() ), { .drawAsDiameter = true } );
        params.tasks->push_back( { std::shared_ptr<void>{}, &radiusTask_ } ); // A non-owning `shared_ptr`.
    }
}

MR_REGISTER_RENDER_OBJECT_IMPL( PlaneObject, RenderPlaneFeatureObject )
RenderPlaneFeatureObject::RenderPlaneFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    static constexpr std::array<Vector3f, 4> cornerPoints = {
        Vector3f( 1, 1, 0 ),
        Vector3f( 1, -1, 0 ),
        Vector3f( -1, -1, 0 ),
        Vector3f( -1, 1, 0 ),
    };

    static const auto mesh = []{
        Triangulation t{ { VertId( 0 ), VertId( 2 ), VertId( 1 ) }, { VertId( 0 ), VertId( 3 ), VertId( 2 ) } };
        return std::make_shared<Mesh>( Mesh::fromTriangles( { cornerPoints.begin(), cornerPoints.end() }, t ) );
    }();
    RenderFeatureMeshComponent<true>::getMesh().setMesh( mesh );

    // Subfeatures.
    getPoints().setPointCloud( std::make_shared<PointCloud>() );
    getLines().setPolyline( std::make_shared<Polyline3>() );
    addSubfeatures( PlaneObject{}, &getLines(), &getPoints() );

    { // Some manual decorations.
        // The square contour.
        getLines().varPolyline()->addFromPoints( cornerPoints.data(), cornerPoints.size(), true );
    }

    nameUiPointIsRelativeToBoundingBoxCenter = false;
    nameUiScreenOffset = Vector2f( 0, 0.1f );
}

std::string RenderPlaneFeatureObject::getObjectNameString( const VisualObject& object, ViewportId viewportId ) const
{
    if ( object.getVisualizeProperty( FeatureVisualizePropertyType::DetailsOnNameTag, viewportId ) )
    {
        Vector3f normal = object.xf().A.col( 2 ).normalized();
        if ( object.parent() )
            normal = object.parent()->worldXf().A * normal;
        constexpr int precision = 2;

        return fmt::format( "{}{}N {:.{}f}, {:.{}f}, {:.{}f}", RenderObjectCombinator::getObjectNameString( object, viewportId ), nameExtrasSeparator, normal.x, precision, normal.y, precision, normal.z, precision );
    }
    else
    {
        return RenderObjectCombinator::getObjectNameString( object, viewportId );
    }
}

MR_REGISTER_RENDER_OBJECT_IMPL( SphereObject, RenderSphereFeatureObject )
RenderSphereFeatureObject::RenderSphereFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object ), object_( &object )
{
    static const auto mesh = []{
        constexpr float radius = 1.0f;
        return std::make_shared<Mesh>( makeSphere( { radius, sphereDetailLevel } ) );
    }();
    getMesh().setMesh( mesh );

    // Subfeatures.
    getPoints().setPointCloud( std::make_shared<PointCloud>() );
    addSubfeatures( SphereObject{}, nullptr, &getPoints() );

    // More or less an arbitrary direction. Just something that's not +X to avoid overlaps with other stuff.
    Vector3f nameTagDir = Vector3f( -1, -1, 0 ).normalized();

    nameUiPoint = nameTagDir;
    nameUiPointIsRelativeToBoundingBoxCenter = false;
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
    nameUiRotateToScreenPlaneAroundSphereCenter = Vector3f( 0, 0, 0 );
}

void RenderSphereFeatureObject::renderUi( const UiRenderParams& params )
{
    RenderObjectCombinator::renderUi( params );

    if ( object_->getVisualizeProperty( DimensionsVisualizePropertyType::diameter, params.viewportId ) )
    {
        radiusTask_ = RenderDimensions::RadiusTask( params, object_->worldXf(), object_->getFrontColor( object_->isSelected() ), { .drawAsDiameter = true, .isSpherical = true } );
        params.tasks->push_back( { std::shared_ptr<void>{}, &radiusTask_ } ); // A non-owning `shared_ptr`.
    }
}

MR_REGISTER_RENDER_OBJECT_IMPL( CylinderObject, RenderCylinderFeatureObject )
RenderCylinderFeatureObject::RenderCylinderFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object ), object_( &object )
{
    static const auto mesh = []{
        constexpr float radius = 1.0f;
        constexpr float length = 1.0f;
        return std::make_shared<Mesh>( makeOpenCylinder( radius, -length / 2, length / 2, numCircleSegments ) );
    }();
    getMesh().setMesh( mesh );

    // Subfeatures.
    getPoints().setPointCloud( std::make_shared<PointCloud>() );
    getLines().setPolyline( std::make_shared<Polyline3>() );
    addSubfeatures( CylinderObject{}, &getLines(), &getPoints() );

    // More or less an arbitrary direction. Just something that's not +X to avoid overlaps with other stuff.
    Vector3f nameTagDir = Vector3f( -1, -1, 0 ).normalized();

    nameUiPoint = nameTagDir;
    nameUiPointIsRelativeToBoundingBoxCenter = false;
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
}

void RenderCylinderFeatureObject::renderUi( const UiRenderParams& params )
{
    RenderObjectCombinator::renderUi( params );

    if ( object_->getVisualizeProperty( DimensionsVisualizePropertyType::diameter, params.viewportId ) )
    {
        radiusTask_ = RenderDimensions::RadiusTask( params, object_->worldXf(), object_->getFrontColor( object_->isSelected() ), { .drawAsDiameter = true } );
        params.tasks->push_back( { std::shared_ptr<void>{}, &radiusTask_ } ); // A non-owning `shared_ptr`.
    }

    if ( object_->getVisualizeProperty( DimensionsVisualizePropertyType::length, params.viewportId ) )
    {
        lengthTask_ = RenderDimensions::LengthTask( params, object_->worldXf(), object_->getFrontColor( object_->isSelected() ), { .points = { Vector3f( 0, 0, -0.5f ), Vector3f( 0, 0, 0.5f ) } } );
        params.tasks->push_back( { std::shared_ptr<void>{}, &lengthTask_ } ); // A non-owning `shared_ptr`.
    }
}

MR_REGISTER_RENDER_OBJECT_IMPL( ConeObject, RenderConeFeatureObject )
RenderConeFeatureObject::RenderConeFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object ), object_( &object )
{
    static const auto mesh = []{
        constexpr float radius = 1;
        constexpr float height = 1;
        return std::make_shared<Mesh>( makeOpenCone( radius, 0, height, numCircleSegments ) );
    }();
    getMesh().setMesh( mesh );
    getMesh().setFlatShading( true );

    // Subfeatures.
    getPoints().setPointCloud( std::make_shared<PointCloud>() );
    getLines().setPolyline( std::make_shared<Polyline3>() );
    addSubfeatures( ConeObject{}, &getLines(), &getPoints() );

    // More or less an arbitrary direction. Just something that's not +X to avoid overlaps with other stuff.
    Vector3f nameTagDir = Vector3f( -1, -1, 0 ).normalized();

    nameUiPoint = Vector3f( 0, 0, 1 ) + nameTagDir;
    nameUiPointIsRelativeToBoundingBoxCenter = false;
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
}

void RenderConeFeatureObject::renderUi( const UiRenderParams& params )
{
    RenderObjectCombinator::renderUi( params );

    if ( object_->getVisualizeProperty( DimensionsVisualizePropertyType::diameter, params.viewportId ) )
    {
        radiusTask_ = RenderDimensions::RadiusTask( params, object_->worldXf(), object_->getFrontColor( object_->isSelected() ), { .center = Vector3f( 0, 0, 1 ), .drawAsDiameter = true } );
        params.tasks->push_back( { std::shared_ptr<void>{}, &radiusTask_ } ); // A non-owning `shared_ptr`.
    }

    if ( object_->getVisualizeProperty( DimensionsVisualizePropertyType::angle, params.viewportId ) )
    {
        angleTask_ = RenderDimensions::AngleTask( params, object_->worldXf(), object_->getFrontColor( object_->isSelected() ), { .rays = { Vector3f( 0.5f, 0, 0.5f ), Vector3f( -0.5f, 0, 0.5f ) }, .isConical = true } );
        params.tasks->push_back( { std::shared_ptr<void>{}, &angleTask_ } ); // A non-owning `shared_ptr`.
    }

    if ( object_->getVisualizeProperty( DimensionsVisualizePropertyType::length, params.viewportId ) )
    {
        lengthTask_ = RenderDimensions::LengthTask( params, object_->worldXf(), object_->getFrontColor( object_->isSelected() ), { .points = { Vector3f{}, Vector3f( 0, 0, 1 ) } } );
        params.tasks->push_back( { std::shared_ptr<void>{}, &lengthTask_ } ); // A non-owning `shared_ptr`.
    }
}

}
