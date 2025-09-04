#pragma once

#include "MRMesh/MRFlagOperators.h"
#include "MRViewer/exports.h"
#include "MRViewer/MRImGuiVectorOperators.h"

#include <imgui.h>

#include <cassert>
#include <optional>
#include <span>
#include <variant>

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

enum class TextIcon
{
    diameter,
};

struct TextColor
{
    // If null, resets the color.
    std::optional<ImU32> color;

    TextColor() {}
    TextColor( ImVec4 color ) : color( ImGui::ColorConvertFloat4ToU32( color ) ) {}
    TextColor( ImU32 color ) : color( color ) {}
};

struct TextFont
{
    // If null, resets the font to the default.
    ImFont* font = nullptr;
};

// Represents an arbitrary block of text, possibly with icons, colors, etc.
struct Text
{
    using ElemVar = std::variant<std::string, TextIcon, TextColor, TextFont>;

    struct Elem
    {
        ElemVar var;

        // The minimum element size. Has no effect if smaller than `computedSize`.
        // Here if Y is negative, inherits the line height.
        ImVec2 size = ImVec2( 0, -1 );

        // Alignment. Only makes sense if `size` is set.
        // [0,0] = top-left, [1,1] = bottom-right.
        ImVec2 align;

        // If set, will increase width to the maximum of all elements with the same column id.
        // This affects how `computedSizeWithPadding` is computed.
        int columnId = -1;

        // The computed content size. Read-only. Don't set manually, `update()` sets this.
        mutable ImVec2 computedSize;

        // Read-only, don't set manually. This is typically `max( size, computedSize )`, with additional adjustments.
        mutable ImVec2 computedSizeWithPadding;

        [[nodiscard]] bool hasDefaultParams() const
        {
            return size == ImVec2( 0, -1 ) && align == ImVec2() && columnId == -1;
        }
    };

    struct Line
    {
        std::vector<Elem> elems;

        // The minimum element size. Has no effect if smaller than `computedSize`.
        // Here if X is negative, inherits the text width.
        ImVec2 size = ImVec2( -1, 0 );

        // Alignment. Only makes sense if `size` is set.
        // [0,0] = top-left, [1,1] = bottom-right.
        ImVec2 align;

        // The computed content size. Read-only. Don't set manually, `update()` sets this.
        mutable ImVec2 computedSize;

        // Read-only, don't set manually. This is typically `max( size, computedSize )`, with additional adjustments.
        mutable ImVec2 computedSizeWithPadding;
    };

    std::vector<Line> lines;

    // The text size for alignment. Unlike `size` in other elements below, this isn't the minimum size.
    // If this is zero, `align` will pivot the text around a point, as opposed to in a box (the math for this just happens to work automatically).
    ImVec2 size;

    // Alignment. [0,0] = top-left, [1,1] = bottom-right.
    ImVec2 align;

    // If null, uses the current font.
    // This is here because `update()` needs it too.
    ImFont* defaultFont = nullptr;

    // The computed content size. Read-only. Don't set manually, `update()` sets this.
    mutable ImVec2 computedSize;

    // No `computedSizeWithPadding`, because for the entire `Text` we use `size` exactly, without `max( size, computedSize )`.

    mutable bool dirty = false;

    Text() {}
    Text( const std::string& text ) { addText( text ); }
    Text( std::string_view text ) { addText( text ); }
    Text( const char* text ) { addText( text ); }

    [[nodiscard]] bool isEmpty() const { return lines.empty(); }

    void addLine()
    {
        dirty = true;
        lines.emplace_back();
        lines.back().elems.emplace_back().var = std::string(); // Add an empty string to force some vertical size on empty lines.
    }

    // Adds some string to the text. Automatically splits by newlines.
    MRVIEWER_API void addText( std::string_view text );

    // A low-level function for adding layout elements.
    // For strings prefer `addText()`. This function doesn't automatically split them by newlines, and so on.
    void addElem( Elem elem )
    {
        dirty = true;
        if ( lines.empty() )
            addLine();
        lines.back().elems.emplace_back( std::move( elem ) );
    }

    // Adds any element type of `ElemVar`.
    void add( auto&& elem )
    {
        addElem( { .var = decltype(elem)(elem) } );
    }

    // Recalculate `computedSize` in various places in this class.
    // If `force == false`, only acts if `dirty == true`. In any case, resets the dirty flag.
    MRVIEWER_API void update( bool force = false ) const;

    // Draws the text to the specified draw list. Automatically calls `update()`.
    // If `defaultTextColor` is not specified, takes it from ImGui.
    MRVIEWER_API void draw( ImDrawList& list, float menuScaling, ImVec2 pos, const TextColor& defaultTextColor = {} ) const;
};

// Draws a floating text bubble.
// If `push` is specified, it's normalized and the text is pushed in that direction,
// by the amount necessarily to clear a perpendicular going through the center point.
// If `pivot` is specified, the bubble is positioned according to its size (like in ImGui::SetNextWindowPos):
// { 0, 0 } for top left corner, { 0.5f, 0.5f } for center (default), { 1, 1 } for bottom right corner.
MRVIEWER_API void text( Element elem, float menuScaling, const Params& params, ImVec2 pos, const Text& text,
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

    Text text;
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
MRVIEWER_API void distance( Element elem, float menuScaling, const Params& params, ImVec2 a, ImVec2 b, const Text& text, const DistanceParams& distanceParams = {} );

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
