#include "MRPalette.h"
#include "ImGuiMenu.h"
#include "MRMesh/MRFinally.h"
#include "MRViewer/MRImGuiVectorOperators.h"
#include "MRViewer/MRViewer.h"
#include "imgui_internal.h"
#include "MRViewport.h"
#include "MRUIRectAllocator.h"
#include "MRMesh/MRConfig.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRJson.h"

#include <fstream>
#include <span>
#include <string>

namespace MR
{

using namespace ImGuiMath;

const std::vector<Color> Palette::DefaultColors =
{
    Color( Vector4f { 0.1f, 0.25f, 1.0f, 1.f } ), // almost blue  -> min value
    Color( Vector4f { 0.15f, 0.5f, 0.75f,1.f } ),
    Color( Vector4f { 0.2f, 0.75f, 0.5f, 1.f } ),
    Color( Vector4f { 0.25f, 1.0f, 0.25f,1.f } ), // almost green -> zero value
    Color( Vector4f { 0.5f, 0.75f, 0.2f, 1.f } ),
    Color( Vector4f { 0.75f, 0.5f, 0.15f,1.f } ),
    Color( Vector4f { 1.0f, 0.25f, 0.1f, 1.f } ), // almost red   -> max value
};

const std::vector<Color> & Palette::GreenRedColors()
{
    static const std::vector<Color> colors =
    {
        Color( Vector4f { 0.25f, 1.0f, 0.25f,1.f } ), // almost green -> min value
        Color( Vector4f { 0.5f, 0.75f, 0.2f, 1.f } ),
        Color( Vector4f { 0.75f, 0.5f, 0.15f,1.f } ),
        Color( Vector4f { 1.0f, 0.25f, 0.1f, 1.f } ), // almost red   -> max value
    };
    return colors;
}

Palette::Palette( const std::vector<Color>& colors )
{
    setBaseColors( colors );
    setDiscretizationNumber( int( colors.size() ) );
    resetLabels();
}

void Palette::setBaseColors( const std::vector<Color>& colors )
{
    parameters_.baseColors = colors;
    updateDiscretizatedColors_();
}

void Palette::setRangeMinMax( float min, float max )
{
    setRangeLimits_( { min, max } );
    resetLabels();
}

void Palette::setRangeMinMaxNegPos( float minNeg, float maxNeg, float minPos, float maxPos )
{
    setRangeLimits_( { minNeg, maxNeg, minPos, maxPos } );
    resetLabels();
}

void Palette::setRangeLimits_( const std::vector<float>& ranges )
{
    const bool trueSize = ranges.size() == 2 || ranges.size() == 4;
    assert( trueSize );
    if ( !trueSize )
    {
        spdlog::warn( "Palette: wrong number of limits!" );
        return;
    }

    bool trueOrder = true;
    for ( int i = 0; i < ranges.size() - 1; ++i )
        trueOrder &= ( ranges[i] <= ranges[i + 1] );
    assert( trueOrder );
    if ( !trueOrder )
    {
        spdlog::warn( "Palette: bad value order!" );
        return;
    }

    const bool needUpdateColors = ranges.size() != parameters_.ranges.size();
    parameters_.ranges = ranges;

    if ( needUpdateColors )
        updateDiscretizatedColors_();
    updateLegendLimits_( parameters_.legendLimits );
    resetLabels();
}


void Palette::resetLabels()
{
    if ( useCustomLabels_ )
        updateCustomLabels_();
    else if ( texture_.filter == FilterType::Linear )
        setZeroCentredLabels_();
    else
        setUniformLabels_();
}
void Palette::setCustomLabels( const std::vector<Label>& labels )
{
    customLabels_ = labels;
    updateCustomLabels_();
    useCustomLabels_ = true;
    showLabels_ = true;
}
void Palette::setLabelsVisible( bool visible )
{
    showLabels_ = visible;
}

bool Palette::loadFromJson( const Json::Value& root )
{
    const auto& colors = root["Colors"];
    std::vector<Color> colorsVector;
    if ( colors.isArray() )
    {
        const int colorsSize = int( colors.size() );
        if ( !colorsSize )
            return false;

        colorsVector = std::vector<Color>( colorsSize, Color() );
        for ( int i = 0; i < colorsSize; ++i )
            deserializeFromJson( colors[i], colorsVector[i] );
    }
    else
        return false;

    const auto& ranges = root["Ranges"];
    std::vector<float> bordersVector;
    if ( ranges.isArray() )
    {
        const int bordersSize = int( ranges.size() );
        if ( !bordersSize )
            return false;

        bordersVector = std::vector<float>( bordersSize, 0.f );
        for ( int i = 0; i < bordersSize; ++i )
        {
            const auto& border = ranges[i];
            if ( border.isDouble() )
                bordersVector[i] = float( border.asDouble() );
        }
    }
    else
        return false;

    const auto& discretization = root["Discretization"];
    int discretizationValue;
    if ( discretization.isInt() )
        discretizationValue = discretization.asInt();
    else
        return false;

    const auto& filter = root["Filter"];
    FilterType filterValue;
    if ( filter.isString() )
    {
        std::string str = filter.asString();
        if ( str == "Linear" )
            filterValue = FilterType::Linear;
        else if ( str == "Discrete" )
            filterValue = FilterType::Discrete;
        else
            return false;
    }
    else
        return false;

    setBaseColors( colorsVector );
    setRangeLimits_( bordersVector );
    setDiscretizationNumber( discretizationValue );
    setFilterType( filterValue );
    return true;
}

void Palette::saveCurrentToJson( Json::Value& root ) const
{
    Json::Value colorsArray = Json::arrayValue;
    const std::vector<Color>& colors = texture_.pixels;
    const auto colorsSize = colors.size() / 2; // second half is gray color
    for ( int i = 0; i < colorsSize; ++i )
        serializeToJson( colors[i], colorsArray[i] );
    root["Colors"] = colorsArray;

    Json::Value ranges = Json::arrayValue;
    const int rangesSize = int( parameters_.ranges.size() );
    for ( int i = 0; i < rangesSize; ++i )
        ranges[i] = parameters_.ranges[i];
    root["Ranges"] = ranges;

    root["Discretization"] = parameters_.discretization;

    std::string str = texture_.filter == FilterType::Linear ? "Linear" : "Discrete";
    root["Filter"] = str;
}

void Palette::setZeroCentredLabels_()
{
    useCustomLabels_ = false;
    labels_.clear();

    auto findStep = [] (float min, float max)
    {
        const float range = max - min;

        // minimum and maximum limits for calculating number of labels (closely, but not equal number limits of labels)
        const auto minN = 5.f;
        const auto maxN = 10.f;

        auto step = 1.f;
        //increase step if it is too small
        while ( range / step < minN )
        {
            step /= 5.f;
            if ( range / step > minN )
                break;
            step /= 2.f;
        }
        //decrease step if it is too large
        while ( range / step > maxN )
        {
            step *= 5.f;
            if ( range / step < maxN )
                break;
            step *= 2.f;
        }

        if ( step <= 1e-4f )
            step = 1e-4f;

        return step;
    };

    auto relativePos2LimitedRelativePos = [this] ( float relativePos )
    {
        return ( relativePos - relativeLimits_.min ) / relativeLimits_.diagonal();
    };

    auto fillLabels = [&] ( float min, float max, float posMin, float posMax )
    {
        const float step = findStep( min, max ) * relativeLimits_.diagonal();
        float value = min / step;
        value = std::ceil( value );
        value *= step;

        const float hidenZone = 0.02f;
        // push intermediate values
        while ( value < max )
        {
            const float pos = 1.f - relativePos2LimitedRelativePos( getRelativePos( value ) );
            if ( pos >= posMin + hidenZone && pos <= posMax - hidenZone )
                labels_.push_back( Label( pos, getStringValue( value ) ) );
            value += step;
        }
    };

    if ( parameters_.ranges.size() == 2 )
    {
        labels_.push_back( Label( 1.f - relativePos2LimitedRelativePos( 0.f ), getStringValue(parameters_.ranges[0])));
        labels_.push_back( Label( 1.f - relativePos2LimitedRelativePos( 1.f ), getStringValue( parameters_.ranges.back() ) ) );

        fillLabels( parameters_.ranges[0], parameters_.ranges[1], 0.f, 1.f );
    }
    else
    {
        //positions are shifted +/- 0.02f for avoiding overlapping
        labels_.push_back( Label( 1.f - relativePos2LimitedRelativePos( 0.f ), getStringValue( parameters_.ranges[0] ) ) );
        labels_.push_back( Label( 1.f - relativePos2LimitedRelativePos( 0.5f ) + 0.02f, getStringValue( parameters_.ranges[1] ) ) );
        labels_.push_back( Label( 1.f - relativePos2LimitedRelativePos( 0.5f ) - 0.02f, getStringValue( parameters_.ranges[2] ) ) );
        labels_.push_back( Label( 1.f - relativePos2LimitedRelativePos( 1.f ), getStringValue( parameters_.ranges[3] ) ) );

        //positions are shifted for avoiding overlapping
        fillLabels( parameters_.ranges[2], parameters_.ranges[3], 0.f, 1.f - relativePos2LimitedRelativePos( 0.5f ) - 0.02f );
        fillLabels( parameters_.ranges[0], parameters_.ranges[1], 1.f - relativePos2LimitedRelativePos( 0.5f ) + 0.02f, 1.f );
    }

    sortLabels_( labels_ );
    showLabels_ = true;
}

void Palette::setUniformLabels_()
{
    useCustomLabels_ = false;
    makeUniformLabels_( labels_ );
    showLabels_ = true;
}

void Palette::makeUniformLabels_( std::vector<Palette::Label>& labels ) const
{
    labels.clear();

    const int colorCount = int( texture_.pixels.size() >> 1 ); // only half because remaining colors are all gray;
    if ( legendLimitIndexes_.min < 0 || legendLimitIndexes_.max > colorCount )
        return;

    const int limitDelta = legendLimitIndexes_.max - legendLimitIndexes_.min;
    const int labelCount = limitDelta ? limitDelta + 1 : 2;
    labels.resize( labelCount );
    if ( !limitDelta || ( parameters_.ranges.size() != 2 && parameters_.ranges.size() != 4 ) )
    {
        labels.resize( 2 );
        labels[0].text = getStringValue( float( legendLimitIndexes_.min ) / colorCount * ( parameters_.ranges.back() - parameters_.ranges[0] ) + parameters_.ranges[0] );
        labels[0].value = 1.f;
        labels[1].text = getStringValue( float( legendLimitIndexes_.max ) / colorCount * ( parameters_.ranges.back() - parameters_.ranges[0] ) + parameters_.ranges[0] );
        labels[1].value = 0.f;
    }
    else if ( parameters_.ranges.size() == 2 )
    {
        for ( int i = 0; i < labelCount; ++i )
        {
            labels[i].text = getStringValue( float( i + legendLimitIndexes_.min ) / colorCount * ( parameters_.ranges.back() - parameters_.ranges[0] ) + parameters_.ranges[0] );
            labels[i].value = 1 - float( i ) / limitDelta;
        }
    }
    else if ( parameters_.ranges.size() == 4 )
    {
        int labelIndex = 0;
        const int colorCountHalf = colorCount / 2;
        for ( int i = legendLimitIndexes_.min; i < colorCountHalf + 1; ++i, ++labelIndex )
        {
            labels[labelIndex].text = getStringValue( float( i ) / colorCountHalf * ( parameters_.ranges[1] - parameters_.ranges[0] ) + parameters_.ranges[0] );
            labels[labelIndex].value = 1.f - float( labelIndex ) / limitDelta;
        }
        for ( int i = std::max( legendLimitIndexes_.min, colorCountHalf + 1 ); i < legendLimitIndexes_.max + 1; ++i, ++labelIndex )
        {
            labels[labelIndex].text = getStringValue( float( i - colorCountHalf - 1 ) / colorCountHalf *
                ( parameters_.ranges.back() - parameters_.ranges[2] ) + parameters_.ranges[2] );
            labels[labelIndex].value = 1.f - float( labelIndex ) / limitDelta;
        }
    }

    // Force add the zero label.
    labels.emplace_back( 0.5, "0" ).isZero = true;

    sortLabels_( labels );
}

void Palette::setDiscretizationNumber( int discretization )
{
    assert( discretization > 1 );
    if ( discretization < 2 )
        return;

    histogramDiscr_.reset();

    parameters_.discretization = discretization;
    updateDiscretizatedColors_();
    updateLegendLimitIndexes_();
    resetLabels();
}

void Palette::setFilterType( FilterType type )
{
    texture_.filter = type;
    updateDiscretizatedColors_();
    resetLabels();
}

void Palette::draw( const std::string& windowName, const ImVec2& pose, const ImVec2& size, bool onlyTopHalf )
{
    const auto menu = ImGuiMenu::instance();
    const auto& viewportSize = Viewport::get().getViewportRect();

    const auto style = getStyleVariables_( menu->menu_scaling() );
    const auto maxLabelWidth = getMaxLabelWidth_( onlyTopHalf );
    const ImVec2 windowSizeMin {
        style.windowPaddingA.x + maxLabelWidth + style.labelToColoredRectSpacing + style.minColoredRectWidth + style.windowPaddingB.x,
        2 * ImGui::GetFontSize(),
    };
    const ImVec2 windowSizeMax {
        width( viewportSize ),
        height( viewportSize ),
    };
    ImGui::SetNextWindowSizeConstraints( windowSizeMin, windowSizeMax, &resizeCallback_, ( void* )this );

    auto paletteWindow = ImGui::FindWindowByName( windowName.c_str() );

    if ( paletteWindow )
    {
        auto currentPos = paletteWindow->Pos;
        auto currentSize = paletteWindow->Size;
        constexpr float cornerSize = 50.0f;
        const auto ctx = ImGui::GetCurrentContext();

        if ( ctx &&
             ctx->IO.MouseClickedCount[0] == 2 &&
             ctx->IO.MousePos.x >= currentPos.x &&
             ctx->IO.MousePos.x < currentPos.x + currentSize.x + cornerSize &&
             ctx->IO.MousePos.y >= currentPos.y &&
             ctx->IO.MousePos.y < currentPos.y + currentSize.y
           )
           {
                ctx->IO.MouseClickedCount[0] = 1; // prevent double-click on the corner to change window size
           }
        if ( prevMaxLabelWidth_ == 0.0f )
            prevMaxLabelWidth_ = maxLabelWidth;
        if ( prevMaxLabelWidth_ != maxLabelWidth )
        {
            currentSize.x += ( maxLabelWidth - prevMaxLabelWidth_ );
            ImGui::SetNextWindowSize( currentSize, ImGuiCond_Always );
            currentPos.x -= ( maxLabelWidth - prevMaxLabelWidth_ );
            ImGui::SetNextWindowPos( currentPos, ImGuiCond_Always );
            prevMaxLabelWidth_ = maxLabelWidth;
        }
    }
    ImGui::BeginSavedWindowPos( windowName, &isWindowOpen_, { size, &pose, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBackground } );

    MR_FINALLY{ ImGui::End(); };

    // allows to move window
    const ImVec2 windowPos = ImGui::GetWindowPos();
    const ImVec2 windowSize = ImGui::GetWindowSize();

    const ImVec2 bgPaddingA = round( ImVec2( 2, 2 ) * menu->menu_scaling() );
    const ImVec2 bgPaddingB = round( ImVec2( 2, 1 ) * menu->menu_scaling() );
    const float labelHeight = ImGui::GetTextLineHeight() + bgPaddingA.y + bgPaddingB.y;
    if ( showLabels_ && labels_.size() == 0 )
    {
        setMaxLabelCount( int( windowSize.y / labelHeight ) );
        resetLabels();
    }

    draw( ImGui::GetWindowDrawList(), menu->menu_scaling(), windowPos, windowSize, onlyTopHalf );
}

void Palette::draw( ImDrawList* drawList, float scaling, const ImVec2& pos, const ImVec2& size, const Color& labelBgColor, bool onlyTopHalf ) const
{
    const auto style = getStyleVariables_( scaling );
    // The max width of the colored rect.
    const float maxColoredRectWidth = size.x - style.windowPaddingA.x - style.windowPaddingB.x - getMaxLabelWidth_( onlyTopHalf ) - style.labelToColoredRectSpacing;
    // The screen coordinates of the bottom-right corner of the colored rectangle.
    const ImVec2 coloredRectEndPos( pos.x + size.x - style.windowPaddingB.x, pos.y + size.y - style.windowPaddingB.y );
    // The top-left corner.
    const ImVec2 coloredRectPos( coloredRectEndPos.x - ( isHistogramEnabled() ? style.minColoredRectWidth : maxColoredRectWidth ), pos.y + style.windowPaddingA.y );
    // The X coordinate of the right edge of the labels.
    const float labelsRightSideX = coloredRectPos.x - style.labelToColoredRectSpacing;

    // Draw histogram, below the labels.
    if ( isHistogramEnabled() && histogram_.maxEntry > 0 )
    {
        #if 1
        const int numSegmentsPerBucket = 1;
        #else // Further split buckets. Till will be useful if we switch to non-linear interpolation.
        // We split each bucket into smaller segments for pretty interpolation.
        // This is the desired segment size (the max size, it can end up smaller).
        const int maxNumPixelsPerSegment = 4; // Intentionally not multiplying by GUI scale.
        const int numSegmentsPerBucket = std::max( 1, ( std::max( 1, int( coloredRectEndPos.y - coloredRectPos.y ) / numBuckets ) + maxNumPixelsPerSegment - 1 ) / maxNumPixelsPerSegment );
        #endif

        int numBuckets = getNumHistogramBuckets();
        if ( onlyTopHalf )
            numBuckets = (numBuckets + 1) / 2; // Lame, but we have to do this since `onlyTopHalf` is only known here, and not when filling the buckets.


        // How many points in total. The last bucket gets only one instead of `numSegmentsPerBucket`.
        int numPoints = numBuckets * numSegmentsPerBucket - numSegmentsPerBucket + 1;

        const ImVec2 histPos( pos.x + style.windowPaddingA.x, coloredRectPos.y );
        const ImVec2 histEndPos( coloredRectPos.x, coloredRectEndPos.y );
        const ImVec2 histSize = histEndPos - histPos;

        // Returns a value in range 0..1.
        auto getBucketValue = [&]( int bucketIndex ) -> float
        {
            return histogram_.buckets.at( bucketIndex ) / float( histogram_.maxEntry );
        };

        auto getHistPoint = [&]( int pointIndex ) -> ImVec2
        {
            ImVec2 ret;
            ret.y = histPos.y + float( pointIndex ) / ( numPoints - 1 ) * histSize.y;

            int thisBucketIndex = pointIndex / numSegmentsPerBucket;
            int segmentInBucket = pointIndex % numSegmentsPerBucket;

            float value = getBucketValue( thisBucketIndex );
            if ( segmentInBucket > 0 )
            {
                float t = segmentInBucket / float( numSegmentsPerBucket );
                value = value * ( 1 - t ) + getBucketValue( thisBucketIndex + 1 ) * t;
            }

            ret.x = histEndPos.x - value * histSize.x;

            return ret;
        };

        const Color lineColor = SceneColors::get( SceneColors::Labels );
        const Color bgColor = lineColor.scaledAlpha( 0.5f );

        { // First draw the background.
            const ImU32 bgColorInt = bgColor.getUInt32();

            // Temporarily disable antialiasing, as it causes artefacts on some machines. Shouldn't be necesasry anyway.
            const ImDrawListFlags oldFlags = drawList->Flags;
            drawList->Flags &= ~ImDrawListFlags_AntiAliasedFill;
            MR_FINALLY{ drawList->Flags = oldFlags; };

            ImVec2 prevPoint;
            for ( int i = 0; i < numPoints; i++ )
            {
                ImVec2 point = round( getHistPoint( i ) );

                if ( i > 0 && point.y != prevPoint.y )
                {
                    drawList->PathLineTo( point );
                    drawList->PathLineTo( prevPoint );
                    drawList->PathLineTo( ImVec2( histEndPos.x, prevPoint.y ) );
                    drawList->PathLineTo( ImVec2( histEndPos.x, point.y ) );
                    drawList->PathFillConvex( bgColorInt );
                }

                prevPoint = point;
            }
        }

        { // Then draw the line.
            const float lineWidth = 2 * scaling;

            for ( int i = 0; i < numPoints; i++ )
                drawList->PathLineTo( getHistPoint( i ) );
            drawList->PathStroke( lineColor.getUInt32(), 0, lineWidth );
        }
    }

    const std::vector<Color>& colors = texture_.pixels;
    const auto sz = colors.size() >> 1; // only half because remaining colors are all gray

    // Draw the colored rectangle.
    switch ( texture_.filter )
    {
    case FilterType::Discrete:
        {
            const ImVec2 percentageBgPaddingA = round( ImVec2( 1, 1 ) * scaling );
            const ImVec2 percentageBgPaddingB = round( ImVec2( 1, 0 ) * scaling );
            const float percentageBgRounding = 2 * scaling;

            auto yStep = size.y / ( legendLimitIndexes_.max - legendLimitIndexes_.min );
            const int indexBegin = int( sz ) - legendLimitIndexes_.max;
            const int indexEnd = int( sz ) - legendLimitIndexes_.min;
            const float legendLimitShift = indexBegin * yStep;
            if ( onlyTopHalf )
                yStep *= 2;
            for ( int i = indexBegin; i < indexEnd; i++ )
            {
                // The positions of the two corners of the colored rect.
                ImVec2 posA( coloredRectPos.x, coloredRectPos.y - legendLimitShift + i * yStep );
                // Here clamp the end Y, otherwise it goes beyond the window when the distance mode is set to unsigned.
                // Normally the user can't see this anyway, because ImGui clamps the rendering to the window, but it matters for the position
                //   of the percentage text, that we draw below.
                ImVec2 posB( coloredRectEndPos.x, std::min( coloredRectPos.y - legendLimitShift + ( i + 1 ) * yStep, coloredRectEndPos.y ) );

                // The rect itself.
                drawList->AddRectFilled( posA, posB, colors[sz - 1 - i].getUInt32() );

                // The percentage of distances in this bucket.
                if ( isDiscretizationPercentagesEnabled() )
                {
                    const std::string text = fmt::format( "{:.1f}%", histogramDiscr_.buckets.at( i ) / float( histogramDiscr_.numEntries ) * 100 );
                    const ImVec2 textSize = ImGui::CalcTextSize( text.c_str() );
                    const ImVec2 textPos = round( posA + ( posB - posA - textSize ) / 2 );

                    // The text background.
                    drawList->AddRectFilled( textPos - percentageBgPaddingA, textPos + textSize + percentageBgPaddingB, Color( 255, 255, 255, 96 ).getUInt32(), percentageBgRounding );

                    // The text itself.
                    drawList->AddText( textPos, Color( 0, 0, 0, 255 ).getUInt32(), text.c_str() );
                }
            }
        }
        break;

    case FilterType::Linear:
        {
            const float scale = 1.f / relativeLimits_.diagonal();
            const float startPos = coloredRectPos.y - size.y * ( 1.f - relativeLimits_.max ) * scale;
            auto yStep = size.y / ( sz - 1 ) * scale;
            if ( onlyTopHalf )
                yStep *= 2;
            for ( int i = 0; i + 1 < sz; i++ )
            {
                const auto color1 = colors[sz - 1 - i].getUInt32();
                const auto color2 = colors[sz - 2 - i].getUInt32();
                drawList->AddRectFilledMultiColor(
                    { coloredRectPos.x, startPos + i * yStep },
                    { coloredRectEndPos.x, startPos + ( i + 1 ) * yStep },
                    color1, color1, color2, color2
                );
            }
        }
        break;
    }

    // Draw labels.
    // After the colored rectangle, because sometimes they overlap (for the histogram, the percentages of out-of-bounds values).
    if ( showLabels_ )
    {
        const ImVec2 bgPaddingA = round( ImVec2( 2, 2 ) * scaling );
        const ImVec2 bgPaddingB = round( ImVec2( 2, 1 ) * scaling );
        const float bgRounding = 2 * scaling;

        const float labelHeight = ImGui::GetTextLineHeight() + bgPaddingA.y + bgPaddingB.y;

        auto pixRange = size.y - labelHeight;
        if ( onlyTopHalf )
            pixRange *= 2;

        for ( int i = 0; i < labels_.size(); ++i )
        {
            std::string textStorage;

            if ( !onlyTopHalf || labels_[i].value <= 0.5f )
            {
                const char* text = getAdjustedLabelText_( i, onlyTopHalf, textStorage );
                if ( !text )
                    continue;

                float textW = ImGui::CalcTextSize( text ).x;
                const ImVec2 textPos = round( ImVec2( labelsRightSideX - textW, coloredRectPos.y + bgPaddingA.y + labels_[i].value * pixRange ) );


                // Append the percentage to the first and last labels.
                if ( isHistogramEnabled() )
                {
                    const bool isFirstHistLabel = i == 0;
                    // This one is disabled if `onlyTopHalf == true`. In that case nothing should be appended to this label anyway,
                    //   but just in case we're also disabling the bool here. (Sync with what `getAdjustedLabelText()` does.)
                    const bool isLastHistLabel = !isFirstHistLabel && !onlyTopHalf && i + 1 == labels_.size();

                    auto appendPercentage = [&]( std::string& out, int n )
                    {
                        out += " : ";

                        int p = int( std::round( n / float( histogram_.numEntries ) * 100 ) );
                        if (p == 0)
                        {
                            out += "<1%";
                            return;
                        }

                        fmt::format_to(std::back_inserter(out), "{}%", p);
                    };

                    if ( isFirstHistLabel )
                    {
                        if ( histogram_.beforeBucket > 0 )
                        {
                            if ( text != textStorage.c_str() )
                                textStorage = text;
                            appendPercentage( textStorage, histogram_.beforeBucket );
                            text = textStorage.c_str();
                            textW = ImGui::CalcTextSize( text ).x;
                        }
                    }
                    else if ( isLastHistLabel )
                    {
                        if ( histogram_.afterBucket > 0 )
                        {
                            if ( text != textStorage.c_str() )
                                textStorage = text;
                            appendPercentage( textStorage, histogram_.afterBucket );
                            text = textStorage.c_str();
                            textW = ImGui::CalcTextSize( text ).x;
                        }
                    }
                }

                { // The background rectangle.
                    // Avoid tiny clipping by the window borders.
                    drawList->PushClipRectFullScreen();
                    MR_FINALLY{ drawList->PopClipRect(); };

                    drawList->AddRectFilled(
                        textPos - bgPaddingA,
                        textPos + ImVec2( textW, ImGui::GetTextLineHeight() ) + bgPaddingB,
                        labelBgColor.getUInt32(),
                        bgRounding,
                        0
                    );
                }

                // The text itself.
                drawList->AddText(
                    textPos,
                    ImGui::GetColorU32( SceneColors::get( SceneColors::Labels ).getUInt32() ),
                    text
                );
            }
        }
    }
}

void Palette::draw( ImDrawList* drawList, float scaling, const ImVec2& pos, const ImVec2& size, bool onlyTopHalf ) const
{
    draw( drawList, scaling, pos, size, getBackgroundColor_().scaledAlpha( 0.75f ), onlyTopHalf );
}

Color Palette::getColor( float val ) const
{
    assert( val >= 0.f && val <= 1.f );

    // only the first row represents the actual palette colours; see `Palette::updateDiscretizatedColors_' for more info
    const std::span<const Color> colors( texture_.pixels.data(), texture_.resolution.x );
    if ( val == 1.f )
        return colors.back();

    float dIdx = val * ( colors.size() - 1 );

    if ( texture_.filter == FilterType::Discrete )
        return colors[int( std::round( dIdx ) )];

    if ( texture_.filter == FilterType::Linear )
    {
        int dId = int( trunc( dIdx ) );
        float c = dIdx - dId;
        return  ( 1 - c ) * colors[dId] + c * colors[dId + 1];
    }

    // unknown FilterType
    return Color();
}

VertColors Palette::getVertColors( const VertScalars& values, const VertBitSet& region, const VertBitSet* valids, const VertBitSet* validsForStats )
{
    MR_TIMER;

    VertColors result( region.find_last() + 1, getInvalidColor() );
    const auto& validRegion = valids ? *valids : region;
    BitSetParallelFor( validRegion, [&result, &values, this] ( VertId v )
    {
        result[v] = getColor( std::clamp( getRelativePos( values[v] ), 0.f, 1.f ) );
    } );

    updateStats( values, region, predFromBitSet( validsForStats ? validsForStats : valids ) );
    return result;
}

float Palette::getRelativePos( float val ) const
{
    if ( parameters_.ranges.size() == 2 )
    {
        const float range = parameters_.ranges[1] - parameters_.ranges[0];
        if ( range != 0.f )
            return ( val - parameters_.ranges[0] ) / range;
        else if ( val < parameters_.ranges[0] )
            return 0.f;
        else if ( val > parameters_.ranges[1] )
            return 1.f;
        else
            return 0.5f;
    }
    else if ( parameters_.ranges.size() == 4 )
    {
        const float centralZoneAbsRange = parameters_.ranges[2] - parameters_.ranges[1];
        const bool isInCentralZone = val >= parameters_.ranges[1] && val <= parameters_.ranges[2];
        if ( isInCentralZone && ( texture_.filter == FilterType::Linear || centralZoneAbsRange <= 0.0f ) )
                return 0.5f;

        float outerZoneRelativeRange = 0.5f;
        float centerZoneRelativeMax = 0.5f;
        if ( texture_.filter == FilterType::Discrete )
        {
            auto realDiscretization = ( 2 * parameters_.discretization + 1 );
            outerZoneRelativeRange = float( parameters_.discretization ) / realDiscretization;
            centerZoneRelativeMax = float( parameters_.discretization + 1 ) / realDiscretization;

            if ( isInCentralZone )
            {
                float centralZoneRelativeRange = 1.0f / realDiscretization;
                float centralZoneRelativeMin = outerZoneRelativeRange;
                return ( val - parameters_.ranges[1] ) / centralZoneAbsRange * centralZoneRelativeRange + centralZoneRelativeMin;
            }
        }

        if ( val < parameters_.ranges[1] )
        {
            const float lowerZoneAbsRange = parameters_.ranges[1] - parameters_.ranges[0];
            if ( lowerZoneAbsRange != 0 )
                return ( val - parameters_.ranges[0] ) / lowerZoneAbsRange * outerZoneRelativeRange;
            else if ( val < parameters_.ranges[0] )
                return 0.f;
            else
                return outerZoneRelativeRange / 2.f;

        }
        else //  val > parameters_.ranges[2]
        {
            const float upperZoneAbsRange = parameters_.ranges[3] - parameters_.ranges[2];
            if ( upperZoneAbsRange != 0 )
                return ( val - parameters_.ranges[2] ) / upperZoneAbsRange * outerZoneRelativeRange + centerZoneRelativeMax;
            else if ( val >= parameters_.ranges[3] )
                return 1.f;
            else
                return outerZoneRelativeRange / 2.f + centerZoneRelativeMax;
        }
    }
    return 0.5f;
}

VertPredicate Palette::predFromBitSet( const VertBitSet* bits )
{
    return bits ? [bits]( VertId v ) { return bits->test( v ); } : VertPredicate{};
}

VertUVCoords Palette::getUVcoords( const VertScalars & values, const VertBitSet & region, const VertPredicate & valids, const VertPredicate & validsForStats )
{
    MR_TIMER;

    VertUVCoords res;
    res.resizeNoInit( region.size() );
    BitSetParallelFor( region, [&] ( VertId v )
    {
        res[v] = getUVcoord( values[v], contains( valids, v ) );
    } );

    updateStats( values, region, validsForStats ? validsForStats : valids );
    return res;
}

void Palette::updateDiscretizatedColors_()
{
    std::vector<Color>& colors = texture_.pixels;
    if (texture_.filter == FilterType::Linear)
    {
        colors = parameters_.baseColors;
    }
    else if ( parameters_.ranges.size() == 4 )
    {
        const auto realDiscretization = parameters_.discretization * 2 + 1;
        colors.resize( realDiscretization );
        for ( int i = 0; i < realDiscretization; ++i )
            colors[i] = getBaseColor_( float( i ) / ( realDiscretization - 1 ) );
    }
    else
    {
        colors.resize( parameters_.discretization );
        for ( int i = 0; i < parameters_.discretization; ++i )
            colors[i] = getBaseColor_( float( i ) / ( parameters_.discretization - 1 ) );
    }

    // add second layer with gray color for invalid values
    const auto sz = colors.size();
    colors.resize( 2 * sz, getInvalidColor() );
    texture_.resolution = { int( sz ), 2 };

    // for FilterType::Discrete, start and end are at the boundary of texels to have equal distance between all colors
    if ( texture_.filter == FilterType::Linear )
    {
        // start and end are in the middle of texels with pure colors
        texStart_ = 0.5f / sz;
        texEnd_ = 1.0f - 0.5f / sz;
    }
    else
    {
        texStart_ = 0;
        texEnd_ = 1;
    }
}

Color Palette::getBaseColor_( float val )
{
    if ( val <= 0 )
        return parameters_.baseColors[0];
    if ( val >= 1 )
        return parameters_.baseColors.back();

    float dIdx = val * ( parameters_.baseColors.size() - 1 );

    int dId = int( trunc( dIdx ) );
    float c = dIdx - dId;
    return  ( 1.f - c ) * parameters_.baseColors[dId] + c * parameters_.baseColors[dId + 1];
}

const Color& Palette::getBackgroundColor_() const
{
    return getViewerInstance().viewport().getParameters().backgroundColor;
}

void Palette::updateCustomLabels_()
{
    labels_ = customLabels_;
    for ( auto& label : labels_ )
    {
        label.value = 1.f - getRelativePos( label.value );
    }
    sortLabels_( labels_ );
}

void Palette::sortLabels_( std::vector<Label>& labels ) const
{
    std::sort( labels.begin(), labels.end(), []( auto itL, auto itR )
    {
        return itL.value < itR.value;
    } );
}

void Palette::updateLegendLimits_( const MinMaxf& limits )
{
    if ( limits.min == limits.max || limits.min >= parameters_.ranges.back() || limits.max <= parameters_.ranges[0] )
        parameters_.legendLimits = {};
    else
        parameters_.legendLimits = limits;

    if ( parameters_.legendLimits.valid() )
        relativeLimits_ = MinMaxf( getRelativePos( parameters_.legendLimits.min ), getRelativePos( parameters_.legendLimits.max ) ).intersection( { 0.f, 1.f } );
    else
        relativeLimits_ = { 0.f, 1.f };

    updateLegendLimitIndexes_();
}

void Palette::updateLegendLimitIndexes_()
{
    const int colorCount = int( texture_.pixels.size() >> 1 ); // only half because remaining colors are all gray
    const int indexMin = int( std::floor( getRelativePos( parameters_.legendLimits.min ) * colorCount ) );
    const int indexMax = int( std::ceil( getRelativePos( parameters_.legendLimits.max ) * colorCount ) );
    if ( parameters_.legendLimits.valid() )
        legendLimitIndexes_ = MinMaxi( indexMin, indexMax ).intersection( { 0, colorCount } );
    else
        legendLimitIndexes_ = { 0, colorCount };
}

const char* Palette::getAdjustedLabelText_( std::size_t labelIndex, bool onlyTopHalf, std::string& storage ) const
{
    if ( !onlyTopHalf && labels_[labelIndex].isZero )
        return nullptr; // Skip zero.

    // Prepend the `<`/`>` signs to the first and last label, when the histogram is enabled.
    if ( isHistogramEnabled() && ( labelIndex == 0 || ( !onlyTopHalf && labelIndex + 1 == labels_.size() ) ) && ( labelIndex == 0 ? histogram_.beforeBucket : histogram_.afterBucket ) > 0 )
    {
        storage.clear();
        storage += labelIndex == 0 ? "> " : "< ";
        storage += labels_[labelIndex].text;

        return storage.c_str();
    }

    return labels_[labelIndex].text.c_str();
}

float Palette::getMaxLabelWidth_( bool onlyTopHalf ) const
{
    float maxLabelWidth = 0.0f;
    std::string textStorage;
    for ( std::size_t i = 0; i < labels_.size(); i++ )
    {
        const char* text = getAdjustedLabelText_( i, onlyTopHalf, textStorage );
        if ( !text )
            continue;
        auto textSize = ImGui::CalcTextSize( text ).x;
        if ( textSize > maxLabelWidth )
            maxLabelWidth = textSize;
    }
    return maxLabelWidth;
}

Palette::StyleVariables Palette::getStyleVariables_( float scaling ) const
{
    const auto& style = ImGui::GetStyle();
    return {
        .windowPaddingA = { style.WindowPadding.x, 0 },
        .windowPaddingB = { style.WindowPadding.x, 0 },
        .labelToColoredRectSpacing = style.FramePadding.x,
        .minColoredRectWidth = 43.0f * scaling,
    };
}

void Palette::resizeCallback_( ImGuiSizeCallbackData* data )
{
    // Currently this seems to be bugged and is called every frame.

    auto palette = ( Palette* )data->UserData;
    if ( !palette )
        return;

    palette->setMaxLabelCount( int( ImGui::GetWindowSize().y / ImGui::GetFontSize() ) );
    palette->resetLabels();
}


std::string Palette::getStringValue( float value ) const
{
    bool needExp = !parameters_.ranges.empty();
    if ( needExp )
    {
        auto rangeDiff = std::abs( parameters_.ranges.back() - parameters_.ranges.front() );
        needExp = rangeDiff != 0.0f && ( rangeDiff > 1e4f || rangeDiff < 1e-2f );
    }

    return valueToString<LengthUnit>( value, {
        .unitSuffix = false,
        .style = needExp ? NumberStyle::exponential : getDefaultUnitParams<LengthUnit>().style,
        .stripTrailingZeroes = false,
    } );
}

int Palette::getMaxLabelCount()
{
    return maxLabelCount_;
}

void Palette::setMaxLabelCount( int val )
{
    maxLabelCount_ = val;
}

void Palette::setLegendLimits( const MinMaxf& limits )
{
    if ( limits == parameters_.legendLimits )
        return;

    updateLegendLimits_( limits );
    resetLabels();
}

int Palette::getNumHistogramBuckets() const
{
    return int( histogram_.buckets.size() );
}

void Palette::setNumHistogramBuckets( int n )
{
    histogram_.reset();
    histogram_.buckets.resize( std::size_t( n ) );
}

int Palette::getDefaultNumHistogramBuckets() const
{
    return 127; // Odd, to have a bucket for zero in the middle.
}

void Palette::updateStats( const VertScalars& values, const VertBitSet& region, const VertPredicate& vertPredicate )
{
    histogram_.reset();
    histogramDiscr_.reset();

    if ( !isHistogramEnabled() && !isDiscretizationPercentagesEnabled() )
        return;

    if ( isDiscretizationPercentagesEnabled() )
    {
        // Update the size of the histogram tracking per-color percentages.
        // Can't use `parameters_.discretization` here, because the actual number of colors doesn't match that when the "central zone" mode is enabled.
        // And I'm told `pixels.size() / 2` is the intended way to calculate that (`/ 2` because half of the texture is gray).
        histogramDiscr_.buckets.resize( texture_.pixels.size() / 2 );
    }

    for ( VertId v : region )
    {
        if ( vertPredicate && !vertPredicate( v ) )
            continue;

        // Invert the position to have larger distances first, to match how UI is displayed.
        float value = 1.f - getRelativePos( values[v] );

        histogram_.addValue( value );
        histogramDiscr_.addValue( value );
    }

    histogram_.finalize();
    histogramDiscr_.finalize();
}

std::vector<Palette::Label> Palette::createUniformLabels() const
{
    std::vector<Palette::Label> result;
    makeUniformLabels_( result );
    return result;
}

void Palette::Histogram::reset()
{
    needsUpdate = true;

    // Zero the buckets.
    std::size_t size = buckets.size();
    buckets.clear();
    buckets.resize( size );

    beforeBucket = 0;
    afterBucket = 0;

    numEntries = 0;
    maxEntry = 0;
}

void Palette::Histogram::addValue( float value )
{
    if ( buckets.empty() )
        return; // This histogram is disabled.

    int bucketIndex = int( std::floor( value * buckets.size() ) );

    if ( bucketIndex < 0 )
        beforeBucket++;
    else if (bucketIndex >= buckets.size())
        afterBucket++;
    else
        buckets[bucketIndex]++;

    numEntries++;

    // `maxEntry` is updated by `finialize()`.
}

void Palette::Histogram::finalize()
{
    needsUpdate = false;

    if ( buckets.empty() )
        maxEntry = 0;
    else
        maxEntry = *std::max_element( buckets.begin(), buckets.end() );
}

Palette::Label::Label( float val, std::string text_ )
{
    value = val;
    text = std::move( text_ );
}

const std::vector<std::string>& PalettePresets::getPresetNames()
{
    return instance_().names_;
}

bool PalettePresets::loadPreset( const std::string& name, Palette& palette )
{
    std::error_code ec;
    auto path = getPalettePresetsFolder();
    if ( !std::filesystem::is_directory( path, ec ) )
    {
        spdlog::warn( "PalettePresets: directory \"{}\" not found!", utf8string( path ) );
        if ( ec )
            spdlog::warn( "PalettePresets: error: \"{}\"", systemToUtf8( ec.message() ) );
        return false;
    }

    path /= asU8String( name ) + u8".json";
    if ( !std::filesystem::is_regular_file( path, ec ) )
    {
        spdlog::error( "PalettePresets: file \"{}\" not found!", utf8string( path ) );
        if ( ec )
            spdlog::warn( "PalettePresets: error: \"{}\"", systemToUtf8( ec.message() ) );
        return false;
    }

    auto res = deserializeJsonValue( path );
    if ( !res )
    {
        spdlog::error( "PalettePresets: deserialize json failed: {}", res.error() );
        return false;
    }

    return palette.loadFromJson( res.value() );
}

Expected<void> PalettePresets::savePreset( const std::string& name, const Palette& palette )
{
    Json::Value root;
    palette.saveCurrentToJson( root );

    std::error_code ec;
    auto path = getPalettePresetsFolder();
    if ( !std::filesystem::is_directory( path, ec ) && !std::filesystem::create_directories( path, ec ) )
    {
        spdlog::error( "PalettePresets: directory \"{}\" does not exist and cannot be created", utf8string( path ) );
        if ( ec )
            spdlog::warn( "PalettePresets: error: \"{}\"", systemToUtf8( ec.message() ) );
        return unexpected( "Cannot save preset with name: \"" + name + "\"" );
    }

    path /= asU8String( name ) + u8".json";
    if ( !serializeJsonValue( root, path ) )
        return unexpected( "Cannot save preset with name: \"" + name + "\"" );

    instance_().update_();
    return {};
}


std::filesystem::path PalettePresets::getPalettePresetsFolder()
{
    return getUserConfigDir() / "PalettePresets";
}

PalettePresets::PalettePresets()
{
    update_();
}

void PalettePresets::update_()
{
    names_.clear();

    auto userPalettesDir = getPalettePresetsFolder();

    std::error_code ec;
    if ( !std::filesystem::is_directory( userPalettesDir, ec ) )
    {
        spdlog::warn( "PalettePresets: directory \"{}\" not found", utf8string( userPalettesDir ) );
        if ( ec )
            spdlog::warn( "PalettePresets: error: \"{}\"", systemToUtf8( ec.message() ) );
        return;
    }

    for ( auto entry : Directory{ userPalettesDir, ec } )
    {
        if ( entry.is_regular_file( ec ) )
        {
            auto ext = entry.path().extension().u8string();
            for ( auto& c : ext )
                c = ( char ) tolower( c );

            if ( ext != u8".json" )
                break;
            names_.push_back( utf8string( entry.path().stem() ) );
        }
    }
    if ( ec )
        spdlog::warn( "PalettePresets: error: \"{}\"", systemToUtf8( ec.message() ) );
}

PalettePresets& PalettePresets::instance_()
{
    static PalettePresets instance;
    return instance;
}

}
