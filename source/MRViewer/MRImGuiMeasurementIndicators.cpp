#include "MRImGuiMeasurementIndicators.h"

#include "MRMesh/MRFinally.h"
#include "MRMesh/MRString.h"
#include "MRViewer/MRColorTheme.h"
#include "MRViewer/MRRibbonFontManager.h"
#include "MRViewer/MRUIStyle.h"

namespace MR::ImGuiMeasurementIndicators
{

using namespace ImGuiMath;

static void expandTriangle( ImVec2& a, ImVec2& b, ImVec2& c, float value )
{
    if ( value > 0 )
    {
        ImVec2 da = normalize( b - a );
        ImVec2 db = normalize( c - b );
        ImVec2 dc = normalize( a - c );
        float fa = std::abs( dot( dc, rot90( da ) ) );
        float fb = std::abs( dot( da, rot90( db ) ) );
        float fc = std::abs( dot( db, rot90( dc ) ) );

        // If this is false, we have a degenerate triangle.
        if ( fa != 0 && fb != 0 && fc != 0 )
        {
            a += ( dc - da ) / fa * value;
            b += ( da - db ) / fb * value;
            c += ( db - dc ) / fc * value;
        }
    }
}

template <typename F>
static void forEachElement( Element elem, F&& func )
{
    if ( bool( elem & Element::outline ) )
        func( Element::outline );
    if ( bool( elem & Element::main ) )
        func( Element::main );
}

Params::Params()
{
    colorMain = Color( 1.f, 1.f, 1.f, 1.f );
    colorOutline = Color( 0.f, 0.f, 0.f, 0.65f );

    colorText = colorMain;
    colorTextOutline = colorOutline;

    bool isDark = ColorTheme::getPreset() == ColorTheme::Preset::Dark;
    if ( !isDark )
    {
        // Swap text color and text outline color, but preserve alpha.
        std::swap( colorText.r, colorTextOutline.r );
        std::swap( colorText.g, colorTextOutline.g );
        std::swap( colorText.b, colorTextOutline.b );
    }

    float tHovered = 0.2f;
    float tActive = 0.3f;
    colorTextOutlineHovered = colorTextOutline * ( 1 - tHovered ) + colorText * tHovered;
    colorTextOutlineActive = colorTextOutline * ( 1 - tActive ) + colorText * tActive;

    static constexpr Stipple::Segment stippleSegmentsDashed[] = {
        { 0, 0.45f },
    };
    stippleDashed = {
        .patternLength = 16,
        .segments = stippleSegmentsDashed,
    };
}

void point( Element elem, const Params& params, ImVec2 point )
{
    forEachElement( elem, [&]( Element thisElem )
    {
        float radius = params.pointDiameter / 2;

        if ( thisElem == Element::outline )
            radius += params.outlineWidth;

        radius *= UI::scale();

        params.list->AddCircleFilled( point, radius, ( thisElem == Element::main ? params.colorMain : params.colorOutline ).getUInt32() );
    } );
}

static Text::FontFunc& defaultFontFuncStorage()
{
    // We default to a monospaced font. Just in case, use a function and re-get it every time.
    static Text::FontFunc ret = []
    {
        return RibbonFontManager::getFontAndSizeByTypeStatic( RibbonFontManager::FontType::Monospace );
    };
    return ret;
}

const Text::FontFunc& Text::getStaticDefaultFontFunc()
{
    return defaultFontFuncStorage();
}

void Text::setStaticDefaultFontFunc( FontFunc func )
{
    defaultFontFuncStorage() = std::move( func );
}

void Text::addText( std::string_view text )
{
    bool first = true;
    split( text, "\n", [&]( std::string_view part )
    {
        if ( first )
            first = false;
        else
            addLine();

        if ( part.empty() )
            return false;

        // If we have an existing text element we can safely merge this with, do that.
        if ( !lines.empty() && !lines.back().elems.empty() && lines.back().elems.back().hasDefaultParams() )
        {
            if ( auto str = std::get_if<std::string>( &lines.back().elems.back().var ) )
            {
                *str += text;
                return false;
            }
        }

        add( std::string( part ) );
        return false;
    } );
}

void Text::update( bool force ) const
{
    if ( !force && !dirty )
        return; // Nothing to do.
    dirty = false;

    std::pair<ImFont*, float> curFont = defaultFont;

    // Compute `elem.computedSize` and `line.computedSize[WithPadding].y`.
    for ( const Line& line : lines )
    {
        line.computedSize.y = 0;

        for ( const Elem& elem : line.elems )
        {
            std::visit( overloaded{
                [&]( const std::string& str )
                {
                    if ( curFont.first )
                        ImGui::PushFont( curFont.first, curFont.second );
                    MR_FINALLY{
                        if ( curFont.first )
                            ImGui::PopFont();
                    };

                    elem.computedSize = round( ImGui::CalcTextSize( str.c_str() ) );
                },
                [&]( TextIcon )
                {
                    // Just this for now.
                    elem.computedSize.x = elem.computedSize.y = std::round( ImGui::GetTextLineHeight() );
                },
                [&]( const TextColor& )
                {
                    elem.computedSize = ImVec2();
                },
                [&]( const TextFont& font )
                {
                    elem.computedSize = ImVec2();
                    curFont = { font.font, font.size };
                },
            }, elem.var );

            if ( elem.computedSize.y > line.computedSize.y )
                line.computedSize.y = elem.computedSize.y;
        }

        line.computedSizeWithPadding.y = std::max( line.computedSize.y, line.size.y );
    }

    float columnWidths[32]{};

    // Compute `elem.computedSizeWithPadding` (here Y requires knowing line heights, and X could've been computed earlier).
    // Also compute initial `computedSizeWithPadding` for elements, before the column alignment.
    for ( const Line& line : lines )
    {
        for ( const Elem& elem : line.elems )
        {
            elem.computedSizeWithPadding = max( elem.size, elem.computedSize );
            if ( elem.size.y < 0 )
                elem.computedSizeWithPadding.y = line.computedSizeWithPadding.y;

            // Store column widths.
            if ( elem.columnId >= 0 )
            {
                bool columnIdOk = elem.columnId < std::size( columnWidths );
                assert( columnIdOk );
                if ( columnIdOk )
                    columnWidths[elem.columnId] = std::max( columnWidths[elem.columnId], elem.computedSizeWithPadding.x );
            }
        }
    }

    // Apply column widths, compute `elem.computedSizeWithPadding`, `line.computedSize.x`, and `computedSize`.
    computedSize = ImVec2();
    for ( const Line& line : lines )
    {
        line.computedSize.x = 0;

        for ( const Elem& elem : line.elems )
        {
            if ( elem.columnId >= 0 )
                elem.computedSizeWithPadding.x = columnWidths[elem.columnId];

            line.computedSize.x += elem.computedSizeWithPadding.x;
        }

        if ( line.computedSize.x > computedSize.x )
            computedSize.x = line.computedSize.x; // Here we don't care about `line.computedSizeWithPadding.x`, as the maximum will be the same anyway.

        computedSize.y += line.computedSizeWithPadding.y;
    }

    // Compute `line.computedSizeWithPadding.x`.
    for ( const Line& line : lines )
    {
        if ( line.size.x < 0 )
            line.computedSizeWithPadding.x = computedSize.x;
        else
            line.computedSizeWithPadding.x = std::max( line.size.x, line.computedSize.x );
    }
}

Text::DrawResult Text::draw( ImDrawList& list, ImVec2 pos, const TextColor& defaultTextColor ) const
{
    DrawResult ret;

    update();

    ImU32 defaultColorFixed = defaultTextColor.color ? *defaultTextColor.color : ImGui::ColorConvertFloat4ToU32( ImGui::GetStyleColorVec4( ImGuiCol_Text ) );
    ImU32 curColor = defaultColorFixed;

    // Specifically for the full text size, we don't do `max( size, computedSize )`, as it makes more sense this way.
    // See the comment on `size` in the class.
    ImVec2 curPos = pos + ( size - computedSize ) * align;
    ret.cornerA = curPos;
    ret.cornerB = curPos + computedSize;

    std::pair<ImFont*, float> curFont = defaultFont;

    for ( const Line& line : lines )
    {
        ImVec2 linePos = curPos + ( line.computedSizeWithPadding - line.computedSize ) * line.align;
        curPos.y += line.computedSizeWithPadding.y;

        for ( const Elem& elem : line.elems )
        {
            ImVec2 elemPos = linePos + ( elem.computedSizeWithPadding - elem.computedSize ) * elem.align;
            linePos.x += elem.computedSizeWithPadding.x;

            std::visit( overloaded{
                [&]( const std::string& str )
                {
                    if ( curFont.first )
                        ImGui::PushFont( curFont.first, curFont.second );
                    MR_FINALLY{
                        if ( curFont.first )
                            ImGui::PopFont();
                    };

                    list.AddText( elemPos, curColor, str.c_str() );
                },
                [&]( TextIcon icon )
                {
                    switch ( icon )
                    {
                    case TextIcon::diameter:
                        list.AddCircle( elemPos + elem.computedSize / 2, elem.computedSize.x / 2 - 2 * UI::scale(), curColor, 0, 1.1f * UI::scale() );
                        list.AddLine(
                            elemPos + ImVec2( elem.computedSize.x - 1.5f, 0.5f ) - ImVec2( 0.5f, 0.5f ),
                            elemPos + ImVec2( 1.5f, elem.computedSize.y - 0.5f ) - ImVec2( 0.5f, 0.5f ),
                            curColor, 1.1f * UI::scale()
                        );
                        break;
                    }
                },
                [&]( const TextColor& color )
                {
                    curColor = color.color ? *color.color : defaultColorFixed;
                },
                [&]( const TextFont& font )
                {
                    curFont = { font.font, font.size };
                },
            }, elem.var );
        }
    }

    return ret;
}

std::optional<TextResult> text( Element elem, const Params& params, ImVec2 pos, const Text& text, const TextParams& textParams, ImVec2 push, ImVec2 pivot )
{
    if ( ( elem & Element::both ) == Element{} )
        return {}; // Nothing to draw.

    if ( text.isEmpty() )
        return {};

    TextResult ret;

    float textOutlineWidth = params.textOutlineWidth * UI::scale();
    float textOutlineRounding = params.textOutlineRounding * UI::scale();
    float textToLineSpacingRadius = params.textToLineSpacingRadius * UI::scale();
    ImVec2 textToLineSpacingA = params.textToLineSpacingA * UI::scale();
    ImVec2 textToLineSpacingB = params.textToLineSpacingB * UI::scale();

    text.update();
    ret.textCornerA = pos - ( text.computedSize * pivot );

    if ( push != ImVec2{} )
    {
        push = normalize( push );
        ImVec2 point = ImVec2( push.x > 0 ? ret.textCornerA.x - textToLineSpacingA.x : ret.textCornerA.x + text.computedSize.x + textToLineSpacingB.x, push.y > 0 ? ret.textCornerA.y - textToLineSpacingA.y : ret.textCornerA.y + text.computedSize.y + textToLineSpacingB.y );
        ret.textCornerA += push * (-dot( push, point - pos ) + textToLineSpacingRadius );
    }

    ret.textCornerA = round( ret.textCornerA );
    ret.textCornerB = ret.textCornerA + text.computedSize;
    ret.bgCornerA = ret.textCornerA - textToLineSpacingA - textOutlineWidth;
    ret.bgCornerB = ret.textCornerB + textToLineSpacingB + textOutlineWidth;

    auto drawLine = [&]( Element lineElem )
    {
        if ( !textParams.line )
            return; // No line specified.

        if ( CompareAll( textParams.line->point ) >= ret.bgCornerA && CompareAll( textParams.line->point ) <= ret.bgCornerB )
            return; // The line end point is already inside the rect.

        ImVec2 point = ( ret.bgCornerA + ret.bgCornerB ) / 2;
        ImVec2 delta = textParams.line->point - point;

        // For now we don't correctly handle the little rounded corner cutouts. This should be barely noticeable.

        float t = std::min(
            delta.x == 0 ? FLT_MAX : ( ( delta.x > 0 ? ret.bgCornerB.x : ret.bgCornerA.x ) - point.x ) / delta.x,
            delta.y == 0 ? FLT_MAX : ( ( delta.y > 0 ? ret.bgCornerB.y : ret.bgCornerA.y ) - point.y ) / delta.y
        );
        assert( t > 0 && t <= 1 );

        auto lineBody = textParams.line->body;
        if ( !lineBody.colorOverride && textParams.borderColor.a > 0 )
            lineBody.colorOverride = textParams.borderColor;

        line( lineElem, params, point + delta * t, textParams.line->point, {
            .body = lineBody,
            .capA = { .decoration = LineCapDecoration::none }, // Could also use `noOutline`, but I think that looks worse.
            .capB = { .decoration = textParams.line->capDecoration },
        } );
    };

    if ( bool( elem & Element::outline ) )
    {
        drawLine( Element::outline );

        const auto& color = textParams.isActive ? params.colorTextOutlineActive : textParams.isHovered ? params.colorTextOutlineHovered : params.colorTextOutline;
        params.list->AddRectFilled( ret.bgCornerA, ret.bgCornerB, color.getUInt32(), textOutlineRounding );
    }
    if ( bool( elem & Element::main ) )
    {
        drawLine( Element::main );

        text.draw( *params.list, ret.textCornerA, params.colorText.getUInt32() );

        // I think the colored frame should be in `Element::main`.
        if ( textParams.borderColor.a > 0 )
        {
            float lineWidthUnselected = params.clickableLabelLineWidth * UI::scale();
            float lineWidthSelected = params.clickableLabelLineWidthSelected * UI::scale();

            float lineWidth = textParams.isSelected ? lineWidthSelected : lineWidthUnselected;
            float outlineWidth = params.clickableLabelOutlineWidth * UI::scale() * 2 + lineWidth;

            float rectShrink = lineWidthUnselected / 2;
            if ( textParams.isSelected )
                rectShrink -= ( lineWidthSelected - lineWidthUnselected ) / 2;

            // First, the outline for the frame.
            // Using `PathRect()` here because it doesn't shrink the rect by half a pixel, unlike `AddRect()`.
            params.list->PathRect( ret.bgCornerA + rectShrink, ret.bgCornerB - rectShrink, textOutlineRounding - rectShrink );
            params.list->PathStroke( params.colorOutline.getUInt32(), ImDrawFlags_Closed, outlineWidth );

            // The frame itself.
            params.list->PathRect( ret.bgCornerA + rectShrink, ret.bgCornerB - rectShrink, textOutlineRounding - rectShrink );
            params.list->PathStroke( textParams.borderColor.scaledAlpha( params.colorOutline.a / 255.f ).getUInt32(), ImDrawFlags_Closed, lineWidth );
        }
    }

    return ret;
}

void arrowTriangle( Element elem, const Params& params, ImVec2 point, ImVec2 dir )
{
    if ( ( elem & Element::both ) == Element{} )
        return; // Nothing to draw.

    float outlineWidth = params.outlineWidth * UI::scale();
    float arrowLen = params.arrowLen * UI::scale();
    float arrowHalfWidth = params.arrowHalfWidth * UI::scale();

    dir = normalize( dir );
    ImVec2 n = rot90( dir );

    ImVec2 a = point;
    ImVec2 b = a - dir * arrowLen + n * arrowHalfWidth;
    ImVec2 c = a - dir * arrowLen - n * arrowHalfWidth;

    if ( bool( elem & Element::outline ) )
    {
        ImVec2 a2 = a;
        ImVec2 b2 = b;
        ImVec2 c2 = c;
        expandTriangle( a2, b2, c2, outlineWidth );

        params.list->AddTriangleFilled( a2, b2, c2, params.colorOutline.getUInt32() );
    }

    if ( bool( elem & Element::main ) )
        params.list->AddTriangleFilled( a, b, c, params.colorMain.getUInt32() );
}

std::optional<LineResult> line( Element elem, const Params& params, ImVec2 a, ImVec2 b, const LineParams& lineParams )
{
    if ( ( elem & Element::both ) == Element{} )
        return {}; // Nothing to draw.

    float arrowLen = params.arrowLen * UI::scale();

    auto midpointsFixed = lineParams.midPoints;

    // Prune `midPoints` that are inside the arrow caps. This improves the arrow cap rendering on curves.
    if ( !midpointsFixed.empty() )
    {
        for ( bool front : { false, true } )
        {
            float capLength = 0;
            switch ( ( front ? lineParams.capB : lineParams.capA ).decoration )
            {
            case LineCapDecoration::none:
            case LineCapDecoration::noOutline:
            case LineCapDecoration::extend:
            case LineCapDecoration::point:
                // Nothing.
                break;
            case LineCapDecoration::arrow:
                capLength = arrowLen;
                break;
            }

            if ( capLength <= 0 )
                continue; // No cap here, nothing to do.

            float remainingLength = capLength;
            ImVec2 curPoint = front ? b : a;

            while ( !midpointsFixed.empty() )
            {
                ImVec2 nextPoint = front ? midpointsFixed.back()/*sic*/ : midpointsFixed.front()/*sic*/;
                float segLengthSq = lengthSq( nextPoint - curPoint );
                if ( segLengthSq >= remainingLength * remainingLength )
                    break; // We're outside of the cap.

                // Pop the point.
                midpointsFixed = midpointsFixed.subspan( front ? 0 : 1, midpointsFixed.size() - 1 );

                curPoint = nextPoint;

                remainingLength -= std::sqrt( segLengthSq );
                if ( remainingLength <= 0 )
                    break; // Just in case.
            }
        }
    }

    if ( a == b && midpointsFixed.empty() )
        return {};

    LineResult ret;

    float lineWidth = ( bool( lineParams.body.flags & LineFlags::narrow ) ? params.smallWidth : params.width ) * UI::scale();
    float outlineWidth = params.outlineWidth * UI::scale();
    float leaderLineLen = params.leaderLineLen * UI::scale();
    float invertedOverhang = params.invertedOverhang * UI::scale();
    float arrowTipBackwardOffset = params.arrowTipBackwardOffset * UI::scale();

    forEachElement( elem, [&]( Element thisElem )
    {
        if ( bool( lineParams.body.flags & LineFlags::onlyOutline ) && thisElem == Element::outline )
            return; // Sic. In that mode we use `Element::main` to draw an outline-colored line.

        ImVec2 points[2] = {a, b};

        // Those are added on the ends of the line, if specified.
        std::optional<ImVec2> extraPoints[2];

        auto midpointsFixed2 = midpointsFixed;

        bool extendOutlineOnCap[2] = { true, true };

        for ( bool front : { false, true } )
        {
            ImVec2& point = points[front];
            std::optional<ImVec2>& extraPoint = extraPoints[front];
            ImVec2 d = front
                ? normalize( b - ( midpointsFixed.empty() ? a : midpointsFixed.back() ) )
                : normalize( a - ( midpointsFixed.empty() ? b : midpointsFixed.front() ) );

            const LineCap& thisCap = front ? lineParams.capB : lineParams.capA;

            // Draw the cap decoration.
            switch ( thisCap.decoration )
            {
            case LineCapDecoration::none:
                // Nothing.
                break;
            case LineCapDecoration::noOutline:
                extendOutlineOnCap[front] = false;
                break;
            case LineCapDecoration::extend:
                point += d * params.notchHalfLen;
                break;
            case LineCapDecoration::arrow:
                {
                    if ( !bool( lineParams.body.flags & LineFlags::noBackwardArrowTipOffset ) && thisCap.text.isEmpty() )
                        point -= d * arrowTipBackwardOffset;
                    ImVec2 arrowTip = point;
                    arrowTriangle( thisElem, params, arrowTip, d );
                    if ( thisCap.text.isEmpty() )
                    {
                        point += d * ( -arrowLen + 1 ); // +1 is to avoid a hairline gap here, we intentionally don't multiply it by `UI::scale()`.

                        // Now trim some extra points to avoid artifacts (which tend to appear when both the stipple and antialiasing are enabled,
                        //   but can probably appear without the stipple too).
                        ImVec2 prevPoint = arrowTip;
                        float accumLen = 0;
                        while ( !midpointsFixed2.empty() )
                        {
                            ImVec2 p = front ? midpointsFixed2.back() : midpointsFixed2.front();
                            if ( dot( p - prevPoint, d ) > 0 )
                                break; // The line has went into a different direction.
                            midpointsFixed2 = midpointsFixed2.subspan( front ? 0 : 1, midpointsFixed2.size() - 1 );
                            accumLen += length( p - prevPoint );
                            if ( accumLen >= arrowLen )
                                break; // The line has probably exited the arrow tip already.
                            prevPoint = p;
                        }
                    }
                    else
                    {
                        point += d * invertedOverhang; // Extend the line instead of shortening it, to prepare for a leader line.
                    }
                }
                break;
            case LineCapDecoration::point:
                ImGuiMeasurementIndicators::point( thisElem, params, point );
                break;
            }

            if ( !thisCap.text.isEmpty() )
            {
                ImVec2 leaderDir( ( d.x > 0 ? 1.f : -1.f ), 0 );
                extraPoint = points[front] + leaderDir * leaderLineLen;
                ( front ? ret.capB : ret.capA ) = text( thisElem, params, *extraPoint, thisCap.text, thisCap.textParams, leaderDir );
            }
        }

        // Check this again, in case we happened to remove more midpoints.
        if ( a == b && midpointsFixed2.empty() )
            return;

        // Those used to extend the outline forward and backward.
        bool isFirstPathPoint = true; // The first point after a flush.
        std::optional<ImVec2> queuedPathPoint;
        std::optional<ImVec2> prevPathPoint;

        // Setting this to false is used for `noOutline` mode on the first cap.
        bool extendOutlineOnFirstPoint = extendOutlineOnCap[0] && ( !lineParams.body.stipple || lineParams.body.stipple->segments.front().a <= 0 );

        auto pathPoint = [&]( ImVec2 p )
        {
            if ( thisElem == Element::main )
            {
                params.list->PathLineTo( p );
                return;
            }

            // At this point we're drawing the outline.

            if ( queuedPathPoint )
            {
                if ( !prevPathPoint || *prevPathPoint != *queuedPathPoint )
                    prevPathPoint = queuedPathPoint;

                if ( isFirstPathPoint && p != *queuedPathPoint )
                {
                    // Extend the first point backwards. (Unless this is the very first point and the cap style is `noOutline`.)

                    if ( extendOutlineOnCap[0] || extendOutlineOnFirstPoint )
                        *queuedPathPoint -= normalize( p - *queuedPathPoint ) * outlineWidth;

                    isFirstPathPoint = false;
                    extendOutlineOnFirstPoint = true; // Reset after the first point.
                }

                params.list->PathLineTo( *queuedPathPoint );
            }
            queuedPathPoint = p;
        };

        // `lastSegment == true` is passed once for the last segment.
        // If the line ends on a stipple gap, then `false` is never passed, which is intentional.
        auto pathStroke = [&]( bool lastSegment )
        {
            if ( thisElem == Element::outline )
            {
                if ( queuedPathPoint && prevPathPoint && ( extendOutlineOnCap[1] || !lastSegment ) )
                {
                    // Extend the last point forward.
                    *queuedPathPoint += normalize( *queuedPathPoint - *prevPathPoint ) * outlineWidth;
                }

                params.list->PathLineTo( *queuedPathPoint );

                isFirstPathPoint = true;
                queuedPathPoint.reset();
                prevPathPoint.reset();
            }

            params.list->PathStroke(
                (
                    thisElem == Element::main
                    ? (
                        bool( lineParams.body.flags & LineFlags::onlyOutline ) ? params.colorTextOutline :
                        lineParams.body.colorOverride ? *lineParams.body.colorOverride :
                        params.colorMain
                    )
                    : params.colorOutline
                ).getUInt32(),
                0,
                lineWidth + ( outlineWidth * 2 ) * ( thisElem == Element::outline )
            );
        };

        auto forEachPoint = [&]( auto&& func )
        {
            if ( extraPoints[0] )
                func( *extraPoints[0] );
            func( points[0] );
            for ( ImVec2 point : midpointsFixed2 )
                func( point );
            func( points[1] );
            if ( extraPoints[1] )
                func( *extraPoints[1] );
        };

        if ( !lineParams.body.stipple )
        {
            forEachPoint( pathPoint );
            pathStroke( true );
        }
        else
        {
            const float patternLen = lineParams.body.stipple->patternLength * UI::scale();

            float t = 0; // The current phase, between 0 and 1.
            bool nowActive = false; // Are we in the middle of a segment right now?
            std::size_t segmIndex = 0; // The index into the pattern.

            // The last known argument to `addPoint()`.
            std::optional<ImVec2> prevPoint;

            auto addPoint = [&]( ImVec2 point )
            {
                if ( prevPoint )
                {
                    // The input segment length measured in pixels.
                    const float inputPixelLen = length( point - *prevPoint );
                    // The input segment length measured in pattern periods. We substract stuff from this.
                    // This may be greater than 1.
                    const float inputPeriodsLen = inputPixelLen / patternLen;

                    // Which part of `inputPeriodsLen` was already consumed. Goes from 0 to `inputPeriodsLen`.
                    float consumedInputPeriodsLen = 0;

                    while ( true )
                    {
                        const Stipple::Segment& thisSegm = lineParams.body.stipple->segments[segmIndex];

                        // How many more periods do we want to skip until the end of the output segment.
                        float remPatternPeriodsLen = thisSegm.get( nowActive ) - t;
                        if ( remPatternPeriodsLen < 0 )
                            remPatternPeriodsLen += 1; // Wrap around.
                        assert( remPatternPeriodsLen >= 0 && remPatternPeriodsLen <= 1 );

                        // Can we finish the output segment during this input segment?
                        if ( inputPeriodsLen - consumedInputPeriodsLen >= remPatternPeriodsLen )
                        {
                            // Yes we can.

                            consumedInputPeriodsLen += remPatternPeriodsLen;

                            // Emit the point.
                            float subT = consumedInputPeriodsLen / inputPeriodsLen;
                            pathPoint( *prevPoint * ( 1.f - subT ) + point * subT );
                            // Update the phase.
                            t = thisSegm.get( nowActive );
                            // Render the output segment if we're finishing it. Update the index into the pattern.
                            if ( nowActive )
                            {
                                // This intentionally uses `lastSegment == false` unconditionally, even if this happens to be the last segment.
                                // That's because `true` only makes sense if we're ending in the middle of the last segment.
                                pathStroke( false );

                                segmIndex++;
                                if ( segmIndex == lineParams.body.stipple->segments.size() )
                                    segmIndex = 0;
                            }
                            // Start/stop the output segment.
                            nowActive = !nowActive;
                        }
                        else
                        {
                            // No, the input segment is too short or the output segment is too long.

                            // Advance the phase.
                            t += inputPeriodsLen - consumedInputPeriodsLen;
                            assert( t >= 0 && t <= 1 ); // This can't possibly wrap around, because otherwise we would just finish the segment.

                            // Emit the intermediate point. This makes segments more smooth.
                            if ( nowActive )
                                pathPoint( point );

                            break;
                        }
                    }
                }

                prevPoint = point;
            };

            forEachPoint( addPoint );

            // Flush the last segment if needed.
            // Here we don't need `pathPoint( *prevPoint );`, because unfinished segments already write the last point,
            //   which is normally used to make them more smooth.
            if ( nowActive )
                pathStroke( true );
        }
    } );

    return ret;
}

std::optional<DistanceResult> distance( Element elem, const Params& params, ImVec2 a, ImVec2 b, const Text& text, const DistanceParams& distanceParams )
{
    if ( ( elem & Element::both ) == Element{} )
        return {}; // Nothing to draw.

    float textToLineSpacingRadius = params.textToLineSpacingRadius * UI::scale();
    ImVec2 textToLineSpacingA = params.textToLineSpacingA * UI::scale();
    ImVec2 textToLineSpacingB = params.textToLineSpacingB * UI::scale();
    float arrowLen = params.arrowLen * UI::scale();
    float totalLenThreshold = params.totalLenThreshold * UI::scale();
    float invertedOverhang = params.invertedOverhang * UI::scale();

    bool useInvertedStyle = lengthSq( b - a ) < totalLenThreshold * totalLenThreshold;
    bool drawTextOutOfLine = useInvertedStyle;

    ImVec2 dir = normalize( b - a );
    ImVec2 n( -dir.y, dir.x );

    ImVec2 center = a + ( b - a ) / 2;

    ImVec2 gapA, gapB;

    // Try to cram the string into the middle of the line.
    if ( !text.isEmpty() && !useInvertedStyle && !distanceParams.moveTextToLineEndIndex )
    {
        text.update();
        ImVec2 textPos = a + ( ( b - a ) - text.computedSize ) / 2.f;

        ImVec2 boxA = textPos - textToLineSpacingA - center;
        ImVec2 boxB = textPos + text.computedSize + textToLineSpacingB - center;
        auto isInBox = [&]( ImVec2 pos ) { return CompareAll( pos ) >= boxA && CompareAll( pos ) <= boxB; };

        if ( isInBox( a ) || isInBox( b ) )
        {
            drawTextOutOfLine = true;
        }
        else
        {
            ImVec2 deltaA = a - center;
            ImVec2 deltaB = b - center;

            for ( ImVec2* delta : { &deltaA, &deltaB } )
            {
                for ( bool axis : { false, true } )
                {
                    if ( (*delta)[axis] < boxA[axis] )
                    {
                        (*delta)[!axis] *= boxA[axis] / (*delta)[axis];
                        (*delta)[axis] = boxA[axis];
                    }
                    else if ( (*delta)[axis] > boxB[axis] )
                    {
                        (*delta)[!axis] *= boxB[axis] / (*delta)[axis];
                        (*delta)[axis] = boxB[axis];
                    }
                }
            }

            gapA = center + deltaA;
            gapB = center + deltaB;

            if ( length( a - gapA ) + length( b - gapB ) < totalLenThreshold + textToLineSpacingRadius * 2 )
            {
                drawTextOutOfLine = true;
            }
            else
            {
                gapA -= dir * textToLineSpacingRadius;
                gapB += dir * textToLineSpacingRadius;
            }
        }
    }

    if ( useInvertedStyle )
    {
        gapA = a - dir * invertedOverhang;
        gapB = b + dir * invertedOverhang;
    }

    DistanceResult ret;

    forEachElement( elem, [&]( Element thisElem )
    {
        if ( !useInvertedStyle && ( text.isEmpty() || drawTextOutOfLine || distanceParams.moveTextToLineEndIndex ) )
        {
            LineParams lineParams{
                .capA = LineCap{ .decoration = LineCapDecoration::arrow },
                .capB = LineCap{ .decoration = LineCapDecoration::arrow },
            };
            if ( distanceParams.moveTextToLineEndIndex )
                ( *distanceParams.moveTextToLineEndIndex ? lineParams.capB : lineParams.capA ).text = text;
            ret.line = line( thisElem, params, a, b, lineParams );
        }
        else
        {
            auto drawLineEnd = [&]( bool front )
            {
                LineParams lineParams{ .capB = LineCap{ .decoration = LineCapDecoration::arrow } };
                if ( useInvertedStyle && distanceParams.moveTextToLineEndIndex && *distanceParams.moveTextToLineEndIndex == front )
                    lineParams.capA.text = text;
                if ( useInvertedStyle )
                    lineParams.body.flags |= LineFlags::noBackwardArrowTipOffset;
                line( thisElem, params, front ? gapB : gapA, front ? b : a, lineParams );
            };

            drawLineEnd( false );
            drawLineEnd( true );

            if ( useInvertedStyle )
                ret.line = line( thisElem, params, a - dir * ( arrowLen / 2 ), b + dir * ( arrowLen / 2 ), { .body = { .flags = LineFlags::narrow } } );
        }

        if ( !distanceParams.moveTextToLineEndIndex )
            ret.text = ImGuiMeasurementIndicators::text( thisElem, params, center, text, distanceParams.textParams, drawTextOutOfLine ? n : ImVec2{} );
    } );

    return ret;
}

} // namespace MR::ImGuiMeasurementIndicators
