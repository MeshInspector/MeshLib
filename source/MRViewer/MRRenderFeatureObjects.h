#pragma once

#include "MRMesh/MRFeatureObject.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRViewer/MRRenderDefaultUiObject.h"
#include "MRViewer/MRRenderLinesObject.h"
#include "MRViewer/MRRenderLinesObject.h"
#include "MRViewer/MRRenderMeshObject.h"
#include "MRViewer/MRRenderPointsObject.h"
#include "MRViewer/MRRenderWrapObject.h"

namespace MR::RenderFeatures
{

struct ObjectParams
{
    float pointSize = 0;
    float pointSizeSub = 0; // For subfeatures.
    float lineWidth = 0;
    float lineWidthSub = 0; // For subfeatures.
    std::uint8_t meshAlpha = 0;
};
[[nodiscard]] MRVIEWER_API const ObjectParams& getObjectParams();

// Returns a nice constrasting color for the current color theme (white for dark theme, black for the light theme).
[[nodiscard]] MRVIEWER_API Color getContrastColor();

// Inherits from a datamodel object, and gives it some properties suitable for rendering secondary parts of features.
// Such a constrating color, automatic visibility based on parent's `FeatureVisualizePropertyType::Subfeatures`.
template <typename Base>
class WrapSecondaryObject : public Base, public RenderWrapObject::BasicWrapperTarget
{
    const ViewportProperty<Color>& getFrontColorsForAllViewports( bool selected = true ) const override
    {
        const_cast<WrapSecondaryObject&>( *this ).setFrontColor( getContrastColor(), selected );
        const_cast<WrapSecondaryObject&>( *this ).setVisibilityMask( target_->getVisualizePropertyMask( FeatureVisualizePropertyType::Subfeatures ) );
        return Base::getFrontColorsForAllViewports( selected );
    }
};

// A common base class for sub-renderobjects that are combined into the proper features.
// `ObjectType` is the underlying datamodel object that stores the mesh, e.g. `ObjectMesh`.
// `RenderObjectType` is the underlying render object, e.g. `RenderMeshObject`.
// If `IsPrimary` is true, the visual properties are copied from the target datamodel object.
template <bool IsPrimary, typename ObjectType, typename RenderObjectType>
using RenderFeatureComponent = RenderWrapObject::Wrapper<std::conditional_t<IsPrimary, RenderWrapObject::CopyVisualProperties<ObjectType>, WrapSecondaryObject<ObjectType>>, RenderObjectType>;

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
        Base::subobject.setPointSize( IsPrimary ? getObjectParams().pointSize : getObjectParams().pointSizeSub );
    }

    auto& getPoints() { return Base::subobject; }
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
        Base::subobject.setLineWidth( IsPrimary ? getObjectParams().lineWidth : getObjectParams().lineWidthSub );
    }

    auto& getLines() { return Base::subobject; }
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
        Base::subobject.setGlobalAlpha( getObjectParams().meshAlpha );
    }

    auto& getMesh() { return Base::subobject; }
};

class RenderPointFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeaturePointsComponent<true>>
{
public:
    MRVIEWER_API RenderPointFeatureObject( const VisualObject& object );
};

class RenderLineFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureLinesComponent<true>>
{
public:
    MRVIEWER_API RenderLineFeatureObject( const VisualObject& object );
};

class RenderCircleFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureLinesComponent<true>, RenderFeaturePointsComponent<false>>
{
public:
    MRVIEWER_API RenderCircleFeatureObject( const VisualObject& object );
};

class RenderPlaneFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureMeshComponent<true>, RenderFeatureLinesComponent<false>, RenderFeaturePointsComponent<false>>
{
public:
    MRVIEWER_API RenderPlaneFeatureObject( const VisualObject& object );
};

class RenderSphereFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureMeshComponent<true>, RenderFeaturePointsComponent<false>>
{
public:
    MRVIEWER_API RenderSphereFeatureObject( const VisualObject& object );
};

class RenderCylinderFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureMeshComponent<true>, RenderFeatureLinesComponent<false>, RenderFeaturePointsComponent<false>>
{
public:
    MRVIEWER_API RenderCylinderFeatureObject( const VisualObject& object );
};

class RenderConeFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureMeshComponent<true>, RenderFeatureLinesComponent<false>, RenderFeaturePointsComponent<false>>
{
public:
    MRVIEWER_API RenderConeFeatureObject( const VisualObject& object );
};

}
