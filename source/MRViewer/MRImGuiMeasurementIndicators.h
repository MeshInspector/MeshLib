#pragma once

#include "MRMesh/MRFlagOperators.h"
#include "MRViewer/exports.h"
#include "MRViewer/MRImGuiVectorOperators.h"

#include <imgui.h>

#include <optional>
#include <span>

namespace MR::ImGuiMeasurementIndicators
{

struct Params
{
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    Color colorMain;
    Color colorOutline;
    Color colorText;
    Color colorTextOutline;

    float pointDiameter = 6;

    float width = 1.5f;
    float smallWidth = 0.75f;
    float outlineWidth = 1.5f;
    float textOutlineWidth = 4.f;
    float textOutlineRounding = 3.f;

    float arrowLen = 12;
    float arrowHalfWidth = 4;
    // The arrow tip is moved back for this amount of pixels,
    //   to compensate for the outline that otherwise makes the tip look longer than it actually is.
    // This only applies if the tip doesn't continue into a line.
    float arrowTipBackwardOffset = 2.5f;

    // The spacing box around the text is extended by this amount.
    ImVec2 textToLineSpacingA = ImVec2( 0, 0 ); // Top-left corner.
    ImVec2 textToLineSpacingB = ImVec2( 0, 2 ); // Bottom-right corner.
    // Further, the lines around the text are shortened by this amount.
    float textToLineSpacingRadius = 8;

    // For a distance, when the total length is less than this, an alternative rendering method is used.
    // If the text is added, its size can make the actual threshold larger.
    float totalLenThreshold = 48;

    // When drawing inverted (short) distances, the line is extended by this amount on both sides.
    float invertedOverhang = 24;

    // The length of the leader lines (short lines attaching text to other things).
    float leaderLineLen = 20;

    // A small perpendicular line at the end of some arrows.
    float notchHalfLen = 8;

    // This picks the colors based on the current color theme.
    MRVIEWER_API Params();
};

enum class Element
{
    main = 1 << 0,
    outline = 1 << 1,
    both = main | outline, // Use this by default.
};
MR_MAKE_FLAG_OPERATORS( Element )

// Draws a point.
MRVIEWER_API void point( Element elem, float menuScaling, const Params& params, ImVec2 point );

enum class StringIcon
{
    none,
    diameter,
};

// A string with an optional custom icon inserted in the middle.
struct StringWithIcon
{
    StringIcon icon{};
    std::size_t iconPos = 0;

    std::string string;

    StringWithIcon() {}

    // Need a bunch of constructors to allow implicit conversions when this is used as a function parameter.
    StringWithIcon( const char* string ) : string( string ) {}
    StringWithIcon( std::string string ) : string( std::move( string ) ) {}

    StringWithIcon( StringIcon icon, std::size_t iconPos, std::string string )
        : icon( icon ), iconPos( iconPos ), string( std::move( string ) )
    {}

    [[nodiscard]] bool isEmpty() const { return icon == StringIcon::none && string.empty(); }

    // Returns the icon width, already scaled to the current scale.
    [[nodiscard]] MRVIEWER_API float getIconWidth() const;
    // Calculates the text size with the icon, using the current font.
    [[nodiscard]] MRVIEWER_API ImVec2 calcTextSize() const;
    // Draws the text with an icon to the specified draw list.
    MRVIEWER_API void draw( ImDrawList& list, float menuScaling, ImVec2 pos, ImU32 color );
};

// Draws a floating text bubble.
// If `push` is specified, it's normalized and the text is pushed in that direction,
// by the amount necessarily to clear a perpendicular going through the center point.
// If `pivot` is specified, the bubble is positioned according to its size (like in ImGui::SetNextWindowPos):
// { 0, 0 } for top left corner, { 0.5f, 0.5f } for center (default), { 1, 1 } for bottom right corner.
MRVIEWER_API void text( Element elem, float menuScaling, const Params& params, ImVec2 pos, StringWithIcon string,
                        ImVec2 push = {}, ImVec2 pivot = { 0.5f, 0.5f } );

// Draws a triangle from an arrow.
MRVIEWER_API void arrowTriangle( Element elem, float menuScaling, const Params& params, ImVec2 point, ImVec2 dir );

struct LineCap
{
    enum class Decoration
    {
        none,
        arrow,
    };
    Decoration decoration{};

    StringWithIcon text;
};

enum class LineFlags
{
    narrow = 1 << 0,
    noBackwardArrowTipOffset = 1 << 1, // Overrides `params.arrowTipBackwardOffset` to zero.
};
MR_MAKE_FLAG_OPERATORS( LineFlags )

struct LineParams
{
    LineFlags flags{};

    LineCap capA{};
    LineCap capB{};

    std::span<const ImVec2> midPoints;
};

// Draws a line or an arrow.
MRVIEWER_API void line( Element elem, float menuScaling, const Params& params, ImVec2 a, ImVec2 b, const LineParams& lineParams = {} );

struct DistanceParams
{
    // If this is set, the text is moved from the middle of the line to one of the line ends (false = A, true = B).
    std::optional<bool> moveTextToLineEndIndex;
};

// Draws a distance arrow between two points, automatically selecting the best visual style.
// The `string` is optional.
MRVIEWER_API void distance( Element elem, float menuScaling, const Params& params, ImVec2 a, ImVec2 b, StringWithIcon string, const DistanceParams& distanceParams = {} );

struct CurveParams
{
    // How many times can be subdivide a curve (resuling in up to 2^depth segments).
    int maxSubdivisionDepth = 10;
    // A curve is always subdivided at least this many times.
    int minSubdivisionDepth = 1;
    // If a curve segment is longer than this, it gets divided in two.
    // You probably don't want to multiply this by `menuScaling`, and we don't do it automatically.
    float subdivisionStepPixels = 4;
};

struct PreparedCurve
{
    ImVec2 a; // The first point.
    ImVec2 b; // The last point.

    [[nodiscard]] ImVec2 endPoint( bool second ) const { return second ? b : a; }

    std::span<ImVec2> midPoints; // All the points in between.
};

// Calculates points for a custom curve, from a function you provide. You can then draw this curve as a `line()`.
//     The points are appended into `pointBuffer` (and for convenience the new points are also returned as `.midPoints`;
//         the first and the last point are not inserted into the vector and are not included in `.midPoints`, they sit in `.a` and `.b`).
//     You should reuse `pointBuffer` between calls for better performance (and `.clear()` it each time you finish drawing the resulting curve,
//         which conveniently preserves vector capacity). Or, if you don't care, just pass an empty vector, and keep it alive as long as you need the curve.
//     `stateA` and `stateB` can have any type, they describe the beginning and the end of the curve respectively. They might often be `0.f` and `1.f`.
//     Let `S` be the type of `stateA`/`stateB` (more on that belw).
//     `stateToPoint` is a lambda `(const S &s) -> ImVec2`. It converts a state into a curve point coordinates.
//     `bisectState` is a lambda `(const S &a, const S &b, int depth) -> S`. Given two states, it produces the state between them. `depth` starts at `0`.
//     `onInsertPoint`, if specified, is a lambda `(ImVec2 point, const S &state) -> void`. It's called right before a point is added into the list.
//     It's actually possible to have multiple state types instead of a single `S`, and template your functions (e.g. to track bisection depth in the type system, etc).
// Example usage:
//     prepareCurve( params, buf, 0, PI*2, [](float x){return std::sin(x);}, [](float a, float b, int /*depth*/){return (a+b)/2;} )
template <typename A, typename B, typename F, typename G, typename H = std::nullptr_t>
[[nodiscard]] PreparedCurve prepareCurve( const CurveParams& curveParams, std::vector<ImVec2>& pointBuffer, const A& stateA, const B& stateB,
    F&& stateToPoint, G&& bisectState, H&& onInsertPoint = nullptr
)
{
    const float pixelStepSq = curveParams.subdivisionStepPixels * curveParams.subdivisionStepPixels;

    std::size_t firstIndex = pointBuffer.size();

    auto makeCurve = [&]( auto makeCurve, int depth, const auto& stateA, const auto& stateB, ImVec2 pointA, ImVec2 pointB ) -> void
    {
        // Check if we need to subdivide.
        if ( depth < curveParams.maxSubdivisionDepth && ( depth < curveParams.minSubdivisionDepth || ImGuiMath::lengthSq( pointB - pointA ) > pixelStepSq ) )
        {
            // Do subdivide.

            auto midState = bisectState( stateA, stateB, int( depth ) ); // A cast to prevent modification.
            ImVec2 midPoint = stateToPoint( midState );

            makeCurve( makeCurve, depth + 1, stateA, midState, pointA, midPoint );
            makeCurve( makeCurve, depth + 1, midState, stateB, midPoint, pointB );
        }
        else
        {
            // No subdivide.

            onInsertPoint( pointB, stateB );
            pointBuffer.push_back( pointB );
        }

    };
    ImVec2 firstPoint = stateToPoint( stateA );
    makeCurve( makeCurve, 0, stateA, stateB, firstPoint, stateToPoint( stateB ) );

    PreparedCurve ret{ .a = firstPoint, .b = pointBuffer.back() };
    pointBuffer.pop_back();
    ret.midPoints = { pointBuffer.data() + firstIndex, pointBuffer.data() + pointBuffer.size() };
    return ret;
}

} // namespace MR::ImGuiMeasurementIndicators
