#include "MRImGuiMeasurementIndicators.h"

#include "MRMesh/MRFinally.h"
#include "MRMesh/MRString.h"
#include "MRViewer/MRColorTheme.h"

#include <parallel_hashmap/phmap.h>

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
}

void point( Element elem, float menuScaling, const Params& params, ImVec2 point )
{
    forEachElement( elem, [&]( Element thisElem )
    {
        float radius = params.pointDiameter / 2;

        if ( thisElem == Element::outline )
            radius += params.outlineWidth;

        radius *= menuScaling;

        params.list->AddCircleFilled( point, radius, ( thisElem == Element::main ? params.colorMain : params.colorOutline ).getUInt32() );
    } );
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

    ImFont* curFont = defaultFont;

    // Compute `elem.computedSize` and `line.computedSize[WithPadding].y`.
    for ( const Line& line : lines )
    {
        line.computedSize.y = 0;

        for ( const Elem& elem : line.elems )
        {
            std::visit( overloaded{
                [&]( const std::string& str )
                {
                    if ( curFont )
                        ImGui::PushFont( curFont );
                    MR_FINALLY{
                        if ( curFont )
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
                    curFont = font.font;
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

void Text::draw( ImDrawList& list, float menuScaling, ImVec2 pos, const TextColor& defaultTextColor ) const
{
    update();

    ImU32 defaultColorFixed = defaultTextColor.color ? *defaultTextColor.color : ImGui::ColorConvertFloat4ToU32( ImGui::GetStyleColorVec4( ImGuiCol_Text ) );
    ImU32 curColor = defaultColorFixed;

    // Specifically for the full text size, we don't do `max( size, computedSize )`, as it makes more sense this way.
    // See the comment on `size` in the class.
    ImVec2 curPos = pos + ( size - computedSize ) * align;

    ImFont* curFont = defaultFont;

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
                    if ( curFont )
                        ImGui::PushFont( curFont );
                    MR_FINALLY{
                        if ( curFont )
                            ImGui::PopFont();
                    };

                    list.AddText( elemPos, curColor, str.c_str() );
                },
                [&]( TextIcon icon )
                {
                    switch ( icon )
                    {
                    case TextIcon::diameter:
                        list.AddCircle( elemPos + elem.computedSize / 2, elem.computedSize.x / 2 - 2 * menuScaling, curColor, 0, 1.1f * menuScaling );
                        list.AddLine(
                            elemPos + ImVec2( elem.computedSize.x - 1.5f, 0.5f ) - ImVec2( 0.5f, 0.5f ),
                            elemPos + ImVec2( 1.5f, elem.computedSize.y - 0.5f ) - ImVec2( 0.5f, 0.5f ),
                            curColor, 1.1f * menuScaling
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
                    curFont = font.font;
                },
            }, elem.var );
        }
    }
}

void text( Element elem, float menuScaling, const Params& params, ImVec2 pos, const Text& text, ImVec2 push, ImVec2 pivot )
{
    if ( ( elem & Element::both ) == Element{} )
        return; // Nothing to draw.

    if ( text.isEmpty() )
        return;

    float textOutlineWidth = params.textOutlineWidth * menuScaling;
    float textOutlineRounding = params.textOutlineRounding * menuScaling;
    float textToLineSpacingRadius = params.textToLineSpacingRadius * menuScaling;
    ImVec2 textToLineSpacingA = params.textToLineSpacingA * menuScaling;
    ImVec2 textToLineSpacingB = params.textToLineSpacingB * menuScaling;

    text.update();
    ImVec2 textPos = pos - ( text.computedSize * pivot );

    if ( push != ImVec2{} )
    {
        push = normalize( push );
        ImVec2 point = ImVec2( push.x > 0 ? textPos.x - textToLineSpacingA.x : textPos.x + text.computedSize.x + textToLineSpacingB.x, push.y > 0 ? textPos.y - textToLineSpacingA.y : textPos.y + text.computedSize.y + textToLineSpacingB.y );
        textPos += push * (-dot( push, point - pos ) + textToLineSpacingRadius );
    }

    if ( bool( elem & Element::outline ) )
        params.list->AddRectFilled( round( textPos ) - textToLineSpacingA - textOutlineWidth, textPos + text.computedSize + textToLineSpacingB + textOutlineWidth, params.colorTextOutline.getUInt32(), textOutlineRounding );
    if ( bool( elem & Element::main ) )
        text.draw( *params.list, menuScaling, round( textPos ), params.colorText.getUInt32() );
}

void arrowTriangle( Element elem, float menuScaling, const Params& params, ImVec2 point, ImVec2 dir )
{
    if ( ( elem & Element::both ) == Element{} )
        return; // Nothing to draw.

    float outlineWidth = params.outlineWidth * menuScaling;
    float arrowLen = params.arrowLen * menuScaling;
    float arrowHalfWidth = params.arrowHalfWidth * menuScaling;

    dir = normalize( dir );
    ImVec2 n = rot90( dir );

    ImVec2 a = point;
    ImVec2 b = a - dir * arrowLen + n * arrowHalfWidth;
    ImVec2 c =  a - dir * arrowLen - n * arrowHalfWidth;

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

void line( Element elem, float menuScaling, const Params& params, ImVec2 a, ImVec2 b, const LineParams& lineParams )
{
    if ( ( elem & Element::both ) == Element{} )
        return; // Nothing to draw.

    float arrowLen = params.arrowLen * menuScaling;

    auto midpointsFixed = lineParams.midPoints;

    // Prune `midPoints` that are inside the arrow caps. This improves the arrow cap rendering on curves.
    if ( !midpointsFixed.empty() )
    {
        for ( bool front : { false, true } )
        {
            float capLength = 0;
            switch ( ( front ? lineParams.capB : lineParams.capA ).decoration )
            {
            case LineCap::Decoration::none:
                // Nothing.
                break;
            case LineCap::Decoration::arrow:
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
        return;

    float lineWidth = ( bool( lineParams.flags & LineFlags::narrow ) ? params.smallWidth : params.width ) * menuScaling;
    float outlineWidth = params.outlineWidth * menuScaling;
    float leaderLineLen = params.leaderLineLen * menuScaling;
    float invertedOverhang = params.invertedOverhang * menuScaling;
    float arrowTipBackwardOffset = params.arrowTipBackwardOffset * menuScaling;

    forEachElement( elem, [&]( Element thisElem )
    {
        ImVec2 points[2] = {a, b};

        // Those are added on the ends of the line, if specified.
        std::optional<ImVec2> extraPoints[2];

        for ( bool front : { false, true } )
        {
            ImVec2& point = points[front];
            std::optional<ImVec2>& extraPoint = extraPoints[front];
            ImVec2 d = front
                ? normalize( b - ( midpointsFixed.empty() ? a : midpointsFixed.back() ) )
                : normalize( a - ( midpointsFixed.empty() ? b : midpointsFixed.front() ) );

            const LineCap& thisCap = front ? lineParams.capB : lineParams.capA;
            switch ( thisCap.decoration )
            {
            case LineCap::Decoration::none:
                // Nothing.
                break;
            case LineCap::Decoration::arrow:
                if ( !bool( lineParams.flags & LineFlags::noBackwardArrowTipOffset ) && thisCap.text.isEmpty() )
                    point -= d * arrowTipBackwardOffset;
                arrowTriangle( thisElem, menuScaling, params, point, d );
                if ( thisCap.text.isEmpty() )
                    point += d * ( -arrowLen + 1 ); // +1 is to avoid a hairline gap here, we intentionally don't multiply it by `menuScaling`.
                else
                    point += d * invertedOverhang; // Extend the line instead of shortening it, to prepare for a leader line.
                break;
            }

            if ( !thisCap.text.isEmpty() )
            {
                ImVec2 leaderDir( ( d.x > 0 ? 1.f : -1.f ), 0 );
                extraPoint = points[front] + leaderDir * leaderLineLen;
                text( thisElem, menuScaling, params, *extraPoint, thisCap.text, leaderDir );
            }

            switch ( thisCap.decoration )
            {
            case LineCap::Decoration::none:
                if ( thisElem == Element::outline )
                    ( extraPoint ? *extraPoint : point ) += ( extraPoint ? normalize( *extraPoint - point ) : d ) * outlineWidth;
                break;
            case LineCap::Decoration::arrow:
                // Nothing.
                break;
            }
        }

        if ( extraPoints[0] )
            params.list->PathLineTo( *extraPoints[0] );
        params.list->PathLineTo( points[0] );
        for ( ImVec2 point : midpointsFixed )
            params.list->PathLineTo( point );
        params.list->PathLineTo( points[1] );
        if ( extraPoints[1] )
            params.list->PathLineTo( *extraPoints[1] );

        params.list->PathStroke( ( thisElem == Element::main ? params.colorMain : params.colorOutline ).getUInt32(), 0, lineWidth + ( outlineWidth * 2 ) * ( thisElem == Element::outline ) );
    } );
}

void distance( Element elem, float menuScaling, const Params& params, ImVec2 a, ImVec2 b, const Text& text, const DistanceParams& distanceParams )
{
    if ( ( elem & Element::both ) == Element{} )
        return; // Nothing to draw.

    float textToLineSpacingRadius = params.textToLineSpacingRadius * menuScaling;
    ImVec2 textToLineSpacingA = params.textToLineSpacingA * menuScaling;
    ImVec2 textToLineSpacingB = params.textToLineSpacingB * menuScaling;
    float arrowLen = params.arrowLen * menuScaling;
    float totalLenThreshold = params.totalLenThreshold * menuScaling;
    float invertedOverhang = params.invertedOverhang * menuScaling;

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

    forEachElement( elem, [&]( Element thisElem )
    {
        if ( !useInvertedStyle && ( text.isEmpty() || drawTextOutOfLine || distanceParams.moveTextToLineEndIndex ) )
        {
            LineParams lineParams{
                .capA = LineCap{ .decoration = LineCap::Decoration::arrow },
                .capB = LineCap{ .decoration = LineCap::Decoration::arrow },
            };
            if ( distanceParams.moveTextToLineEndIndex )
                ( *distanceParams.moveTextToLineEndIndex ? lineParams.capB : lineParams.capA ).text = text;
            line( thisElem, menuScaling, params, a, b, lineParams );
        }
        else
        {
            auto drawLineEnd = [&]( bool front )
            {
                LineParams lineParams{ .capB = LineCap{ .decoration = LineCap::Decoration::arrow } };
                if ( useInvertedStyle && distanceParams.moveTextToLineEndIndex && *distanceParams.moveTextToLineEndIndex == front )
                    lineParams.capA.text = text;
                if ( useInvertedStyle )
                    lineParams.flags |= LineFlags::noBackwardArrowTipOffset;
                line( thisElem, menuScaling, params, front ? gapB : gapA, front ? b : a, lineParams );
            };

            drawLineEnd( false );
            drawLineEnd( true );

            if ( useInvertedStyle )
                line( thisElem, menuScaling, params, a - dir * ( arrowLen / 2 ), b + dir * ( arrowLen / 2 ), { .flags = LineFlags::narrow } );
        }

        if ( !distanceParams.moveTextToLineEndIndex )
            ImGuiMeasurementIndicators::text( thisElem, menuScaling, params, center, text, drawTextOutOfLine ? n : ImVec2{} );
    } );
}

} // namespace MR::ImGuiMeasurementIndicators
