#pragma once

#include "MRMesh/MRAngleMeasurementObject.h"
#include "MRMesh/MRDistanceMeasurementObject.h"
#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRPointMeasurementObject.h"
#include "MRMesh/MRPointMeasurementObject.h"
#include "MRMesh/MRRadiusMeasurementObject.h"
#include "MRViewer/exports.h"
#include "MRViewer/MRRenderDefaultObjects.h"
#include "MRViewer/MRRenderDimensions.h"

namespace MR
{

using RenderDimensionObject = RenderDefaultUiObject;

class RenderPointObject : public RenderDimensionObject
{
    const PointMeasurementObject* object_ = nullptr;
    RenderDimensions::PointTask task_;
public:
    MRVIEWER_API RenderPointObject( const VisualObject& object );
    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;
};

class RenderDistanceObject : public RenderDimensionObject
{
    const DistanceMeasurementObject* object_ = nullptr;
    RenderDimensions::LengthTask task_;
public:
    MRVIEWER_API RenderDistanceObject( const VisualObject& object );
    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;
};

class RenderRadiusObject : public RenderDimensionObject
{
    const RadiusMeasurementObject* object_ = nullptr;
    RenderDimensions::RadiusTask task_;
public:
    MRVIEWER_API RenderRadiusObject( const VisualObject& object );
    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;
};

class RenderAngleObject : public RenderDimensionObject
{
    const AngleMeasurementObject* object_ = nullptr;
    RenderDimensions::AngleTask task_;
public:
    MRVIEWER_API RenderAngleObject( const VisualObject& object );
    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;
};

}
