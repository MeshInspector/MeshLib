#pragma once

#include "MRRenderPointsObject.h"
#include "MRRenderLinesObject.h"
#include "MRRenderMeshObject.h"
#include "MRViewer/MRRenderDefaultUiObject.h"

namespace MR
{

class RenderPointFeatureObject : public RenderDefaultUiMixin<RenderPointsObject>
{
public:
    MRVIEWER_API RenderPointFeatureObject( const VisualObject& object );
};

class RenderLineFeatureObject : public RenderDefaultUiMixin<RenderLinesObject>
{
public:
    MRVIEWER_API RenderLineFeatureObject( const VisualObject& object );
};

class RenderPlaneFeatureObject : public RenderDefaultUiMixin<RenderMeshObject>
{
    using RenderDefaultUiMixin::RenderDefaultUiMixin;
};

class RenderCircleFeatureObject : public RenderDefaultUiMixin<RenderLinesObject>
{
public:
    MRVIEWER_API RenderCircleFeatureObject( const VisualObject& object );
};

class RenderSphereFeatureObject : public RenderDefaultUiMixin<RenderMeshObject>
{
public:
    MRVIEWER_API RenderSphereFeatureObject( const VisualObject& object );
};

class RenderCylinderFeatureObject : public RenderDefaultUiMixin<RenderMeshObject>
{
public:
    MRVIEWER_API RenderCylinderFeatureObject( const VisualObject& object );
};

class RenderConeFeatureObject : public RenderDefaultUiMixin<RenderMeshObject>
{
public:
    MRVIEWER_API RenderConeFeatureObject( const VisualObject& object );
};

}
