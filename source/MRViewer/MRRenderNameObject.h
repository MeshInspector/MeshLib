#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRVector2.h"
#include "MRViewer/exports.h"
#include "MRViewer/MRImGui.h"

#include <optional>

namespace MR
{

class MRVIEWER_CLASS RenderNameObject : public virtual IRenderObject
{
    struct Task : BasicUiRenderTask
    {
        const VisualObject* object = nullptr;
        const UiRenderParams* params = nullptr;

        ImVec2 windowCornerA;
        ImVec2 windowCornerB;

        ImVec2 point;
        ImVec2 point2;
        ImVec2 textCenter;

        std::string text;
        ImVec2 textPos;
        ImVec2 textSize;
        ImVec2 paddingA;
        ImVec2 paddingB;

        bool prevFrameHovered = false;
        bool isHovered = false;
        bool isActive = false;

        MRVIEWER_API void earlyBackwardPass( const BackwardPassParams& backParams ) override;
        MRVIEWER_API void renderPass() override;
    };
    Task task_;
public:
    RenderNameObject( const VisualObject& object ) { task_.object = &object; }

    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;

    MRVIEWER_API virtual std::string getObjectNameString( const VisualObject& object, ViewportId viewportId ) const;

    // The name tag is displayed as a text bubble, attached to a specific point on the model with at most 2-segment line.
    // The first segment offset is specified in 3d model coordinates, and the second offset is in screen coordinates.
    // The offsets can be tiny, since any non-zero offset is automatically extended to make sure the text bubble doesn't overlap the attachment point.

    /// The line attachment point.
    Vector3f nameUiPoint;

    /// If true, `nameUiPoint` is relative to the center of the bounding box. Otherwise it's relative to the origin.
    /// Either way it's in model space.
    bool nameUiPointIsRelativeToBoundingBoxCenter = true;

    /// Which way the name is moved relative to the `point`, in model space. The length is respected.
    Vector3f nameUiLocalOffset;

    /// Which way the name is moved relative to the `point`, in screen space (Y is down). The length is respected.
    /// This is automatically multiplied by the global GUI scale.
    Vector2f nameUiScreenOffset;

    /// If set, the vector from this point to `point` is rotated to be in the screen plane. `localOffset` is also rotated by the same amount.
    /// Good for having the name on a spherical surface.
    std::optional<Vector3f> nameUiRotateToScreenPlaneAroundSphereCenter;

    /// If true, the `localOffset` is rotated 90 degrees CW after projecting it to screen space.
    /// This is great if you have some axis or line that you don't want the name to overlap.
    bool nameUiRotateLocalOffset90Degrees = false;
};

}
