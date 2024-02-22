#pragma once

#include "MRRenderPointsObject.h"
#include "MRRenderLinesObject.h"
#include "MRRenderMeshObject.h"
#include "MRViewer/MRRenderDefaultUiObject.h"

namespace MR
{

class RenderPointFeatureObject : public RenderDefaultUiObject<RenderPointsObject>
{
public:
    MRVIEWER_API RenderPointFeatureObject( const VisualObject& object );
};

class RenderLineFeatureObject : public RenderDefaultUiObject<RenderLinesObject>
{
public:
    MRVIEWER_API RenderLineFeatureObject( const VisualObject& object );
};

// No `class RenderPlaneFeatureObject` for now, because planes look ok with default parameters.
// If you add it, don't forget to add `setupRenderObject_()` to `PlaneObject`, like other features do.

class RenderCircleFeatureObject : public RenderDefaultUiObject<RenderLinesObject>
{
public:
    MRVIEWER_API RenderCircleFeatureObject( const VisualObject& object );
};

class RenderSphereFeatureObject : public RenderDefaultUiObject<RenderMeshObject>
{
public:
    MRVIEWER_API RenderSphereFeatureObject( const VisualObject& object );
};

class RenderCylinderFeatureObject : public RenderDefaultUiObject<RenderMeshObject>
{
public:
    MRVIEWER_API RenderCylinderFeatureObject( const VisualObject& object );
};

class RenderConeFeatureObject : public RenderDefaultUiObject<RenderMeshObject>
{
public:
    MRVIEWER_API RenderConeFeatureObject( const VisualObject& object );
};

}
