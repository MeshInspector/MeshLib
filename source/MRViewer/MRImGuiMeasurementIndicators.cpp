#include "MRImGuiMeasurementIndicators.h"

#include "MRViewer/MRColorTheme.h"

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
    colorOutline = Color( 0.f, 0.f, 0.f, 0.5f );

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

float StringWithIcon::getIconWidth() const
{
    switch ( icon )
    {
    case StringIcon::none:
        return 0;
    case StringIcon::diameter:
        return std::round( ImGui::GetTextLineHeight() );
    }
    assert( false && "Invalid icon enum." );
    return 0;
}

ImVec2 StringWithIcon::calcTextSize() const
{
    return ImGui::CalcTextSize( string.data(), string.data() + string.size() ) + ImVec2( getIconWidth(), 0 );
}

void StringWithIcon::draw( ImDrawList& list, float menuScaling, ImVec2 pos, ImU32 color )
{
    if ( icon == StringIcon{} )
    {
        list.AddText( pos, color, string.data(), string.data() + string.size() );
    }
    else
    {
        assert( iconPos <= string.size() );

        ImVec2 iconPixelPos = pos + ImVec2( ImGui::CalcTextSize( string.data(), string.data() + iconPos ).x, 0 );
        ImVec2 iconPixelSize( getIconWidth(), ImGui::GetTextLineHeight() );

        list.AddText( pos, color, string.data(), string.data() + iconPos );
        list.AddText( iconPixelPos + ImVec2( iconPixelSize.x, 0 ), color, string.data() + iconPos, string.data() + string.size() );

        switch ( icon )
        {
        case StringIcon::none:
            // Nothing, and this should be unreachable.
            break;
        case StringIcon::diameter:
            list.AddCircle( iconPixelPos + iconPixelSize / 2, iconPixelSize.x / 2 - 2 * menuScaling, color, 0, 1.1f * menuScaling );
            list.AddLine(
                iconPixelPos + ImVec2( iconPixelSize.x - 1.5f, 0.5f ) - ImVec2( 0.5f, 0.5f ),
                iconPixelPos + ImVec2( 1.5f, iconPixelSize.y - 0.5f ) - ImVec2( 0.5f, 0.5f ),
                color, 1.1f * menuScaling
            );
            break;
        }
    }
}

void text( Element elem, float menuScaling, const Params& params, ImVec2 pos, StringWithIcon string, ImVec2 push, ImVec2 pivot )
{
    if ( ( elem & Element::both ) == Element{} )
        return; // Nothing to draw.

    if ( string.isEmpty() )
        return;

    float textOutlineWidth = params.textOutlineWidth * menuScaling;
    float textOutlineRounding = params.textOutlineRounding * menuScaling;
    float textToLineSpacingRadius = params.textToLineSpacingRadius * menuScaling;
    ImVec2 textToLineSpacingA = params.textToLineSpacingA * menuScaling;
    ImVec2 textToLineSpacingB = params.textToLineSpacingB * menuScaling;

    ImVec2 textSize = string.calcTextSize();
    ImVec2 textPos = pos - ( textSize * pivot );

    if ( push != ImVec2{} )
    {
        push = normalize( push );
        ImVec2 point = ImVec2( push.x > 0 ? textPos.x - textToLineSpacingA.x : textPos.x + textSize.x + textToLineSpacingB.x, push.y > 0 ? textPos.y - textToLineSpacingA.y : textPos.y + textSize.y + textToLineSpacingB.y );
        textPos += push * (-dot( push, point - pos ) + textToLineSpacingRadius );
    }

    if ( bool( elem & Element::outline ) )
        params.list->AddRectFilled( round( textPos ) - textToLineSpacingA - textOutlineWidth, textPos + textSize + textToLineSpacingB + textOutlineWidth, params.colorTextOutline.getUInt32(), textOutlineRounding );
    if ( bool( elem & Element::main ) )
        string.draw( *params.list, menuScaling, round( textPos ), params.colorText.getUInt32() );
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

void distance( Element elem, float menuScaling, const Params& params, ImVec2 a, ImVec2 b, StringWithIcon string, const DistanceParams& distanceParams )
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
    if ( !string.isEmpty() && !useInvertedStyle && !distanceParams.moveTextToLineEndIndex )
    {
        ImVec2 textSize = string.calcTextSize();
        ImVec2 textPos = a + ( ( b - a ) - textSize ) / 2.f;

        ImVec2 boxA = textPos - textToLineSpacingA - center;
        ImVec2 boxB = textPos + textSize + textToLineSpacingB - center;
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
        if ( !useInvertedStyle && ( string.isEmpty() || drawTextOutOfLine || distanceParams.moveTextToLineEndIndex ) )
        {
            LineParams lineParams{
                .capA = LineCap{ .decoration = LineCap::Decoration::arrow },
                .capB = LineCap{ .decoration = LineCap::Decoration::arrow },
            };
            if ( distanceParams.moveTextToLineEndIndex )
                ( *distanceParams.moveTextToLineEndIndex ? lineParams.capB : lineParams.capA ).text = string;
            line( thisElem, menuScaling, params, a, b, lineParams );
        }
        else
        {
            auto drawLineEnd = [&]( bool front )
            {
                LineParams lineParams{ .capB = LineCap{ .decoration = LineCap::Decoration::arrow } };
                if ( useInvertedStyle && distanceParams.moveTextToLineEndIndex && *distanceParams.moveTextToLineEndIndex == front )
                    lineParams.capA.text = string;
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
            text( thisElem, menuScaling, params, center, string, drawTextOutOfLine ? n : ImVec2{} );
    } );
}

} // namespace MR::ImGuiMeasurementIndicators
