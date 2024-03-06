#include "MRRenderFeatureObjects.h"

#include "MRMesh/MRCircleObject.h"
#include "MRMesh/MRConeObject.h"
#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRFeatures.h"
#include "MRMesh/MRLineObject.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMeshNormals.h"
#include "MRMesh/MRPlaneObject.h"
#include "MRMesh/MRPointObject.h"
#include "MRMesh/MRSphereObject.h"
#include "MRMesh/MRSubfeatures.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPolyline.h"

namespace MR::RenderFeatures
{

static constexpr int numCircleSegments = 128;
static constexpr int sphereDetailLevel = 2048;

const ObjectParams& getObjectParams()
{
    static const ObjectParams ret{
        .pointSize = 10,
        .pointSizeSub = 6,
        .lineWidth = 3,
        .lineWidthSub = 2,
        .meshAlpha = 128,
    };
    return ret;
}

// This is similar to `Features::forEachSubfeature`, but slightly adjusted to be suitable for visualization.
static void forEachVisualSubfeature( const Features::Primitives::Variant& feature, std::function<void( const Features::Primitives::Variant& subfeature )> func )
{
    Features::forEachSubfeature( feature, [&]( const Features::SubfeatureInfo& params )
    {
        if ( !params.isInfinite )
            func( params.create() );
    } );

    std::visit( overloaded{
        [&]( const Features::Primitives::Sphere& sphere )
        {
            (void)sphere;
        },
        [&]( const Features::Primitives::ConeSegment& cone )
        {
            if ( !cone.isCircle() )
            {
                // Cap centers.
                for ( bool negativeCap : { false, true } )
                {
                    if ( std::isfinite( negativeCap ? cone.negativeLength : cone.positiveLength ) &&
                        ( negativeCap ? cone.negativeSideRadius : cone.positiveSideRadius ) > 0 )
                    {
                        func( cone.basePoint( negativeCap ) );
                    }
                }
            }
        },
        [&]( const Features::Primitives::Plane& plane )
        {
            (void)plane;
        },
    }, feature );
}

// Extracts subfeatures from `sourceObject` and writes them out `outputPoints` and `outputLines`.
// `sourceObject` must point to a temporary object of the desired type, with identity xf.
static void addSubfeatures( const VisualObject& sourceObject, ObjectLines* outputLines, ObjectPoints* outputPoints )
{
    // Actually add the subfeatures:

    auto parentFeature = Features::primitiveFromObject( sourceObject );
    assert( parentFeature && "Can't convert this object to a feature" );
    if ( !parentFeature )
        return;

    forEachVisualSubfeature( *parentFeature, [&]( const Features::Primitives::Variant& subFeature ) -> void
    {
        // It's a bit jank to utilize `primitiveToObject()` just to switch over the subfeature types, but it's very convenient.

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
        return fmt::format( "{}\n{:.{}f}, {:.{}f}, {:.{}f}", RenderObjectCombinator::getObjectNameString( object, viewportId ), point.x, precision, point.y, precision, point.z, precision );
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

    nameUiLocalOffset = Vector3f( 0.01f, 0, 0 );
    nameUiRotateLocalOffset90Degrees = true;
}

std::string RenderLineFeatureObject::getObjectNameString( const VisualObject& object, ViewportId viewportId ) const
{
    if ( object.getVisualizeProperty( FeatureVisualizePropertyType::DetailsOnNameTag, viewportId ) )
    {
        Vector3f delta = object.xf().A.col( 0 ) * 2.f;
        if ( object.parent() )
            delta = object.parent()->worldXf().A * delta;
        constexpr int precision = 2;

        // U+0394 GREEK CAPITAL LETTER DELTA
        return fmt::format( "{}\n\xCE\x94 {:.{}f}, {:.{}f}, {:.{}f}", RenderObjectCombinator::getObjectNameString( object, viewportId ), delta.x, precision, delta.y, precision, delta.z, precision );
    }
    else
    {
        return RenderObjectCombinator::getObjectNameString( object, viewportId );
    }
}

MR_REGISTER_RENDER_OBJECT_IMPL( CircleObject, RenderCircleFeatureObject )
RenderCircleFeatureObject::RenderCircleFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
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
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
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
    getMesh().setMesh( mesh );

    // Subfeatures.
    getPoints().setPointCloud( std::make_shared<PointCloud>() );
    getLines().setPolyline( std::make_shared<Polyline3>() );
    addSubfeatures( PlaneObject{}, &getLines(), &getPoints() );

    { // Some manual decorations.
        // The square contour.
        getLines().varPolyline()->addFromPoints( cornerPoints.data(), cornerPoints.size(), true );

        // The normal.
        std::array<Vector3f, 4> normalPoints = {
            Vector3f( 0, 0, 0 ),
            Vector3f( 0, 0, 0.5f ),
        };
        getLines().varPolyline()->addFromPoints( normalPoints.data(), normalPoints.size(), true );
    }

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

        return fmt::format( "{}\nN {:.{}f}, {:.{}f}, {:.{}f}", RenderObjectCombinator::getObjectNameString( object, viewportId ), normal.x, precision, normal.y, precision, normal.z, precision );
    }
    else
    {
        return RenderObjectCombinator::getObjectNameString( object, viewportId );
    }
}

MR_REGISTER_RENDER_OBJECT_IMPL( SphereObject, RenderSphereFeatureObject )
RenderSphereFeatureObject::RenderSphereFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
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
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
    nameUiRotateToScreenPlaneAroundSphereCenter = Vector3f( 0, 0, 0 );
}

MR_REGISTER_RENDER_OBJECT_IMPL( CylinderObject, RenderCylinderFeatureObject )
RenderCylinderFeatureObject::RenderCylinderFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
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
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
}

MR_REGISTER_RENDER_OBJECT_IMPL( ConeObject, RenderConeFeatureObject )
RenderConeFeatureObject::RenderConeFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
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
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
}

}
