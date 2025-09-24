#pragma once

#include "MRMesh/MRColor.h"
#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRVector2.h"
#include "MRViewer/MRRenderClickableRect.h"
#include "MRViewer/exports.h"

#include <optional>

namespace MR
{
class Viewport;
}

namespace MR::RenderDimensions
{

struct Tolerance
{
    float positive = 0; // Should be positive or zero.
    float negative = 0; // Should be negative or zero.
};

struct CommonParams
{
    // What object to select when the label is clicked.
    // Optional. Not clickable if this is null.
    const VisualObject* objectToSelect = nullptr;

    // Optional. If specified, this name is drawn above the measurement.
    std::string objectName;
};

struct PointParams
{
    CommonParams common;

    Vector3f point; // The world point.

    // [1,1] to display the text to the bottom-right of the point, [-1,-1] to display it to the top-left.
    // Typically either X or Y here should be 1 or -1.
    ImVec2 align = ImVec2( 1, 1 );

    // If specified, we're comparing `point` against this position.
    std::optional<Vector3f> referencePoint;

    // Only makes sense if `referencePoint` is set.
    // If specified, we're measuring along this normal. Otherwise we're using eucledian distance.
    // This doesn't need to be normalized.
    Vector3f referenceNormal;

    // Tolerances. Only make sense if `referencePoint` is set.
    // The `.negative` here only makes sense if `referenceNormal` is also set.
    std::optional<Tolerance> tolerance;
};

class PointTask : public BasicClickableRectUiRenderTask
{
    Viewport* viewport_ = nullptr;
    Color color_;
    PointParams params_;

public:
    PointTask() {}

    MRVIEWER_API PointTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const PointParams& params );
    MRVIEWER_API void renderPass() override;

    // Implement `BasicClickableRectUiRenderTask`:
    MRVIEWER_API void onClick() override;
};

struct RadiusParams
{
    CommonParams common;

    // The center point.
    Vector3f center;

    // The length of this is the radius. This is also the preferred drawing direction relative to `center`.
    Vector3f radiusAsVector = Vector3f( 1, 0, 0 );

    // The preferred normal for non-spherical radiuses. The length is ignored, and this is automatically adjusted to be perpendicular to `radiusAsVector`.
    Vector3f normal = Vector3f( 0, 0, 1 );

    // Whether we should draw this as a diameter instead of a radius.
    bool drawAsDiameter = false;

    // Whether this is a sphere radius, as opposed to circle/cylinder radius.
    bool isSpherical = false;

    // The visual leader line length multiplier, relative to the radius.
    // You're recommended to set a min absolute value for the resulting length when rendering.
    float visualLengthMultiplier = 2 / 3.f;
};

class RadiusTask : public BasicClickableRectUiRenderTask
{
    Viewport* viewport_ = nullptr;
    Color color_;
    RadiusParams params_;

public:
    RadiusTask() {}

    MRVIEWER_API RadiusTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const RadiusParams& params );
    MRVIEWER_API void renderPass() override;

    // Implement `BasicClickableRectUiRenderTask`:
    MRVIEWER_API void onClick() override;
};

struct AngleParams
{
    CommonParams common;

    // The center point.
    Vector3f center;

    // The two rays.
    // Use the length of the shorter ray as the arc radius.
    std::array<Vector3f, 2> rays;

    // Whether this is a conical angle. The middle line between the rays is preserved, but the rays themselves can be rotated.
    bool isConical = false;

    // Whether we should draw a ray from the center point to better visualize the angle. Enable this if there isn't already a line object there.
    std::array<bool, 2> shouldVisualizeRay{ true, true };
};

class AngleTask : public BasicClickableRectUiRenderTask
{
    Viewport* viewport_ = nullptr;
    Color color_;
    AngleParams params_;

public:
    AngleTask() {}
    MRVIEWER_API AngleTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const AngleParams& params );
    MRVIEWER_API void renderPass() override;

    // Implement `BasicClickableRectUiRenderTask`:
    MRVIEWER_API void onClick() override;
};


struct LengthParams
{
    CommonParams common;

    // The points between which we're measuring.
    std::array<Vector3f, 2> points;

    // Whether the distance should be displayed as a negative one.
    bool drawAsNegative = false;

    // If set, use only once axis (with this index, 0..2) instead of euclidean.
    std::optional<int> onlyOneAxis;

    // If set, we're comparing the distance with a reference value.
    std::optional<float> referenceValue;

    // Tolerances. Only make sense if `referenceValue` is set.
    std::optional<Tolerance> tolerance;
};

class LengthTask : public BasicClickableRectUiRenderTask
{
    Viewport* viewport_ = nullptr;
    Color color_;
    LengthParams params_;

    // For single-axis measurements, this computes the corner point.
    MRVIEWER_API Vector3f computeCornerPoint();

public:
    LengthTask() {}
    MRVIEWER_API LengthTask( const UiRenderParams& uiParams, const AffineXf3f& xf, Color color, const LengthParams& params );
    MRVIEWER_API void renderPass() override;

    // Implement `BasicClickableRectUiRenderTask`:
    MRVIEWER_API void onClick() override;
};

}
