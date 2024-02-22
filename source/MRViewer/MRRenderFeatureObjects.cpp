#include "MRRenderFeatureObjects.h"

#include "MRMesh/MRCircleObject.h"
#include "MRMesh/MRConeObject.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRLineObject.h"
#include "MRMesh/MRPlaneObject.h"
#include "MRMesh/MRPointObject.h"
#include "MRMesh/MRSphereObject.h"

namespace MR
{

MR_REGISTER_RENDER_OBJECT_IMPL( PointObject, RenderPointFeatureObject )
RenderPointFeatureObject::RenderPointFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    nameUiScreenOffset = Vector2f( 0, 0.1f );
}

MR_REGISTER_RENDER_OBJECT_IMPL( LineObject, RenderLineFeatureObject )
RenderLineFeatureObject::RenderLineFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    nameUiLocalOffset = Vector3f( 0.01f, 0, 0 );
    nameUiRotateLocalOffset90Degrees = true;
}

MR_REGISTER_RENDER_OBJECT_IMPL( CircleObject, RenderCircleFeatureObject )
RenderCircleFeatureObject::RenderCircleFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    // More or less an arbitrary direction. Just something that's not +X to avoid overlaps with other stuff.
    Vector3f nameTagDir = Vector3f( -1, -1, 0 ).normalized();

    nameUiPoint = nameTagDir;
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
}

MR_REGISTER_RENDER_OBJECT_IMPL( SphereObject, RenderSphereFeatureObject )
RenderSphereFeatureObject::RenderSphereFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
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
    // More or less an arbitrary direction. Just something that's not +X to avoid overlaps with other stuff.
    Vector3f nameTagDir = Vector3f( -1, -1, 0 ).normalized();

    nameUiPoint = nameTagDir;
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
}

MR_REGISTER_RENDER_OBJECT_IMPL( ConeObject, RenderConeFeatureObject )
RenderConeFeatureObject::RenderConeFeatureObject( const VisualObject& object )
    : RenderObjectCombinator( object )
{
    // More or less an arbitrary direction. Just something that's not +X to avoid overlaps with other stuff.
    Vector3f nameTagDir = Vector3f( -1, -1, 0 ).normalized();

    nameUiPoint = Vector3f( 0, 0, 1 ) + nameTagDir;
    nameUiLocalOffset = nameTagDir * 2.f / 3.f;
}

}
