#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRObjectWithMeasurements.h"
#include "MRMesh/MRVector2.h"
#include "MRViewer/exports.h"

#include <imgui.h>

#include <optional>

namespace MR
{

class Viewport;

class RenderMeasurementsObject : public virtual IRenderObject
{
    template <MeasurementPropertyEnum Kind>
    struct BasicTask : BasicUiRenderTask
    {
        MeasurementPropertyParameters<Kind> params;
        float menuScaling = 1;
        Viewport* viewport = nullptr;
        Vector3f dirTowardsCamera;
    };

    struct RadiusTask : BasicTask<RadiusVisualizePropertyType>
    {
        MRVIEWER_API void renderPass() override;
    };
    struct AngleTask : BasicTask<AngleVisualizePropertyType>
    {
        MRVIEWER_API void renderPass() override;
    };
    struct LengthTask : BasicTask<LengthVisualizePropertyType>
    {
        MRVIEWER_API void renderPass() override;
    };

    const VisualObject* object = nullptr;
    const IObjectWithMeasurements* interface = nullptr; // Points to the same object as `object`.
public:
    MRVIEWER_API RenderMeasurementsObject( const VisualObject& object );

    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;
};

}
