#include "MRRenderFeatureObjects.h"

#include "MRMesh/MRCircleObject.h"
#include "MRMesh/MRConeObject.h"
#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRLineObject.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMeshNormals.h"
#include "MRMesh/MRPlaneObject.h"
#include "MRMesh/MRPointObject.h"
#include "MRMesh/MRSphereObject.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPolyline.h"

namespace MR
{

static constexpr int numCircleSegments = 128;
static constexpr int sphereDetailLevel = 2048;

const RenderFeatureObjectParams& getRenderFeatureObjectParams()
{
    static const RenderFeatureObjectParams ret{
        .pointSize = 10,
        .lineWidth = 3,
        .meshAlpha = 128,
    };
    return ret;
}

MR_REGISTER_RENDER_OBJECT_IMPL( PointObject, RenderPointFeatureObject )
RenderPointFeatureObject::RenderPointFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    static const auto pointCloud = []{
        auto ret = std::make_shared<PointCloud>();
        ret->points.push_back( Vector3f() );
        ret->validPoints.resize( 1, true );
        return ret;
    }();
    subobject.setPointCloud( pointCloud );

    nameUiScreenOffset = Vector2f( 0, 0.1f );
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

MR_REGISTER_RENDER_OBJECT_IMPL( CircleObject, RenderCircleFeatureObject )
RenderCircleFeatureObject::RenderCircleFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    static const auto polyline = []{
        auto ret = std::make_shared<Polyline3>();
        constexpr int numPoints = 128;
        std::array<Vector3f, numPoints> points;
        for ( int i = 0; i < numPoints; ++i )
        {
            float angle = i * 2 * PI_F / numPoints;
            points[i].x = cosf( angle );
            points[i].y = sinf( angle );
        }
        ret->addFromPoints( points.data(), numPoints, true );
        return ret;
    }();
    subobject.setPolyline( polyline );

    // More or less an arbitrary direction. Just something that's not +X to avoid overlaps with other stuff.
    Vector3f nameTagDir = Vector3f( -1, -1, 0 ).normalized();
    nameUiPoint = nameTagDir;
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
}

MR_REGISTER_RENDER_OBJECT_IMPL( SphereObject, RenderSphereFeatureObject )
RenderSphereFeatureObject::RenderSphereFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    static const auto mesh = []{
        constexpr float radius = 1.0f;
        return std::make_shared<Mesh>( makeSphere( { radius, sphereDetailLevel } ) );
    }();
    subobject.setMesh( mesh );

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
    subobject.setMesh( mesh );

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
    subobject.setMesh( mesh );
    subobject.setFlatShading( true );

    // More or less an arbitrary direction. Just something that's not +X to avoid overlaps with other stuff.
    Vector3f nameTagDir = Vector3f( -1, -1, 0 ).normalized();

    nameUiPoint = Vector3f( 0, 0, 1 ) + nameTagDir;
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
}

}
