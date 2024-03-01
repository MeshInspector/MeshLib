#pragma once

#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRViewer/MRRenderDefaultUiObject.h"
#include "MRViewer/MRRenderLinesObject.h"
#include "MRViewer/MRRenderLinesObject.h"
#include "MRViewer/MRRenderMeshObject.h"
#include "MRViewer/MRRenderPointsObject.h"
#include "MRViewer/MRRenderWrapObject.h"

namespace MR
{

struct RenderFeatureObjectParams
{
    float pointSize = 3;
    float lineWidth = 1;
    std::uint8_t meshAlpha = 128;
};
[[nodiscard]] MRVIEWER_API const RenderFeatureObjectParams& getRenderFeatureObjectParams();

// A common base class for sub-renderobjects that are combined into the proper features.
// `ObjectType` is the underlying datamodel object that stores the mesh, e.g. `ObjectMesh`.
// `RenderObjectType` is the underlying render object, e.g. `RenderMeshObject`.
// If `IsPrimary` is true, the visual properties are copied from the target datamodel object.
template <bool IsPrimary, typename ObjectType, typename RenderObjectType>
using RenderFeatureComponent = RenderWrapObject::Wrapper<std::conditional_t<IsPrimary, RenderWrapObject::CopyVisualProperties<ObjectType>, ObjectType>, RenderObjectType>;

// This renderobject draws custom points.
// If `IsPrimary` is true, the visual properties are copied from the target datamodel object.
template <bool IsPrimary>
class RenderFeaturePointsComponent : public RenderFeatureComponent<IsPrimary, ObjectPoints, RenderPointsObject>
{
    using Base = RenderFeatureComponent<IsPrimary, ObjectPoints, RenderPointsObject>;
public:
    RenderFeaturePointsComponent( const VisualObject& object )
        : Base( object )
    {
        Base::subobject.setPointSize( getRenderFeatureObjectParams().pointSize );
    }
};

// This renderobject draws custom lines.
// If `IsPrimary` is true, the visual properties are copied from the target datamodel object.
template <bool IsPrimary>
class RenderFeatureLinesComponent : public RenderFeatureComponent<IsPrimary, ObjectLines, RenderLinesObject>
{
    using Base = RenderFeatureComponent<IsPrimary, ObjectLines, RenderLinesObject>;
public:
    RenderFeatureLinesComponent( const VisualObject& object )
        : Base( object )
    {
        Base::subobject.setLineWidth( getRenderFeatureObjectParams().lineWidth );
    }
};

// This renderobject draws a custom mesh.
// If `IsPrimary` is true, the visual properties are copied from the target datamodel object.
template <bool IsPrimary>
class RenderFeatureMeshComponent : public RenderFeatureComponent<IsPrimary, ObjectMesh, RenderMeshObject>
{
    using Base = RenderFeatureComponent<IsPrimary, ObjectMesh, RenderMeshObject>;
public:
    RenderFeatureMeshComponent( const VisualObject& object )
        : Base( object )
    {
        Base::subobject.setGlobalAlpha( getRenderFeatureObjectParams().meshAlpha );
    }
};

class RenderPointFeatureObject : public RenderDefaultUiObject<RenderFeaturePointsComponent<true>>
{
public:
    MRVIEWER_API RenderPointFeatureObject( const VisualObject& object );
};

class RenderLineFeatureObject : public RenderDefaultUiObject<RenderFeatureLinesComponent<true>>
{
public:
    MRVIEWER_API RenderLineFeatureObject( const VisualObject& object );
};

// No `class RenderPlaneFeatureObject` for now, because planes look ok with default parameters.
// If you add it, don't forget to add `setupRenderObject_()` to `PlaneObject`, like other features do.

class RenderCircleFeatureObject : public RenderDefaultUiObject<RenderFeatureLinesComponent<true>>
{
public:
    MRVIEWER_API RenderCircleFeatureObject( const VisualObject& object );
};

class RenderSphereFeatureObject : public RenderDefaultUiObject<RenderFeatureMeshComponent<true>>
{
public:
    MRVIEWER_API RenderSphereFeatureObject( const VisualObject& object );
};

class RenderCylinderFeatureObject : public RenderDefaultUiObject<RenderFeatureMeshComponent<true>>
{
public:
    MRVIEWER_API RenderCylinderFeatureObject( const VisualObject& object );
};

class RenderConeFeatureObject : public RenderDefaultUiObject<RenderFeatureMeshComponent<true>>
{
public:
    MRVIEWER_API RenderConeFeatureObject( const VisualObject& object );
};

}
