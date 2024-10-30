#include "MRPalette.h"
#include "ImGuiMenu.h"
#include "imgui_internal.h"
#include "MRViewport.h"
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
}

void Palette::setRangeMinMaxNegPos( float minNeg, float maxNeg, float minPos, float maxPos )
{
    setRangeLimits_( { minNeg, maxNeg, minPos, maxPos } );
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

    auto findStep = []( float min, float max )
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

    auto fillLabels = [&]( float min, float max, float posMin, float posMax )
    {
        const float step = findStep( min, max );
        float value = min / step;
        value = std::ceil( value );
        value *= step;

        // push intermediate values
        while ( value < max )
        {
            const float pos = 1.f - getRelativePos( value );
            if ( pos >= posMin && pos <= posMax )
                labels_.push_back( Label( pos, getStringValue( value ) ) );
            value += step;
        }
    };

    if ( parameters_.ranges.size() == 2 )
    {
        labels_.push_back( Label( 1.f, getStringValue( parameters_.ranges[0] ) ) );
        labels_.push_back( Label( 0.f, getStringValue( parameters_.ranges.back() ) ) );

        fillLabels( parameters_.ranges[0], parameters_.ranges[1], 0.02f, 0.98f );
    }
    else
    {
        //positions are shifted +/- 0.02f for avoiding overlapping
        labels_.push_back( Label( 1.f, getStringValue( parameters_.ranges[0] ) ) );
        labels_.push_back( Label( 0.52f, getStringValue( parameters_.ranges[1] ) ) );
        labels_.push_back( Label( 0.48f, getStringValue( parameters_.ranges[2] ) ) );
        labels_.push_back( Label( 0.f, getStringValue( parameters_.ranges[3] ) ) );

        //positions are shifted for avoiding overlapping
        fillLabels( parameters_.ranges[2], parameters_.ranges[3], 0.02f, 0.46f );
        fillLabels( parameters_.ranges[0], parameters_.ranges[1], 0.54f, 0.98f );
    }

    sortLabels_();
    showLabels_ = true;
}

void Palette::setUniformLabels_()
{
    useCustomLabels_ = false;
    labels_.clear();

    if ( parameters_.ranges.size() == 2 )
    {
        int num = texture_.filter == FilterType::Linear ? 5 : parameters_.discretization + 1;
        if ( maxLabelCount_ && maxLabelCount_ < num )
            num = maxLabelCount_;

        labels_.resize( num );

        for ( int i = 0; i < num; ++i )
        {
            labels_[i].text = getStringValue( float( i ) / ( num - 1 ) * ( parameters_.ranges.back() - parameters_.ranges[0] ) + parameters_.ranges[0] );
            labels_[i].value = 1 - float( i ) / ( num - 1 );
        }
    }
    else
    {
        int num = texture_.filter == FilterType::Linear ? 3 : parameters_.discretization + 1;
        if ( maxLabelCount_ && maxLabelCount_ / 2 < num )
            num = maxLabelCount_ / 2;

        labels_.resize( num * 2 );

        if ( texture_.filter == FilterType::Linear )
        {
            for ( int i = 0; i < num; ++i )
            {
                labels_[i].text = getStringValue( float( i ) / ( num - 1 ) * ( parameters_.ranges[1] - parameters_.ranges[0] ) + parameters_.ranges[0] );
                labels_[i].value = 1 - float( i ) / ( num - 1 ) * 0.5f;
                labels_[i + num].text = getStringValue( float( i ) / ( num - 1 ) * ( parameters_.ranges[3] - parameters_.ranges[2] ) + parameters_.ranges[2] );
                labels_[i + num].value = 0.5f - float( i ) / ( num - 1 ) * 0.5f;
            }
            labels_[num - 1].value += 0.02f;
            labels_[num].value -= 0.02f;
        }
        else
        {
            for ( int i = 0; i < num; ++i )
            {
                labels_[i].text = getStringValue( float( i ) / ( num - 1 ) * ( parameters_.ranges[1] - parameters_.ranges[0] ) + parameters_.ranges[0] );
                labels_[i].value = 1.f - float( i ) / ( num * 2 - 1 );
                labels_[num + i].text = getStringValue( float( i ) / ( num - 1 ) * ( parameters_.ranges[3] - parameters_.ranges[2] ) + parameters_.ranges[2] );
                labels_[num + i].value = 1.f - float( i + num ) / ( num * 2 - 1 );
            }
        }
    }

    sortLabels_();
    showLabels_ = true;
}

void Palette::setDiscretizationNumber( int discretization )
{
    assert( discretization > 1 );
    if ( discretization < 2 )
        return;

    parameters_.discretization = discretization;
    updateDiscretizatedColors_();
}

void Palette::setFilterType( FilterType type )
{
    texture_.filter = type;
    updateDiscretizatedColors_();
    resetLabels();
}

void Palette::draw( const std::string& windowName, const ImVec2& pose, const ImVec2& size, bool onlyTopHalf )
{
    float maxTextSize = 0.0f;
    for ( const auto& label : labels_ )
    {
        auto textSize = ImGui::CalcTextSize( label.text.c_str() ).x;
        if ( textSize > maxTextSize )
            maxTextSize = textSize;
    }

    const auto& style = ImGui::GetStyle();
    const auto menu = ImGuiMenu::instance();
    const auto& windowSize = Viewport::get().getViewportRect();

    ImGui::SetNextWindowPos( pose, ImGuiCond_Appearing );
    ImGui::SetNextWindowSize( size, ImGuiCond_Appearing );

    ImGui::SetNextWindowSizeConstraints( { maxTextSize + style.WindowPadding.x + style.FramePadding.x + 20.0f * menu->menu_scaling(), 2 * ImGui::GetFontSize() }, { width( windowSize ), height( windowSize ) }, &resizeCallback_, ( void* )this );

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
            prevMaxLabelWidth_ = maxTextSize;
        if ( prevMaxLabelWidth_ != maxTextSize )
        {
            currentSize.x += ( maxTextSize - prevMaxLabelWidth_ );
            ImGui::SetNextWindowSize( currentSize, ImGuiCond_Always );
            currentPos.x -= ( maxTextSize - prevMaxLabelWidth_ );
            ImGui::SetNextWindowPos( currentPos, ImGuiCond_Always );
            prevMaxLabelWidth_ = maxTextSize;
        }
    }

    ImGui::Begin( windowName.c_str(), &isWindowOpen_,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBackground );

    // draw gradient palette
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // allows to move window
    const auto& actualPose = ImGui::GetWindowPos();
    const auto& actualSize = ImGui::GetWindowSize();

    if ( showLabels_ )
    {
        if ( labels_.size() == 0 )
        {
            setMaxLabelCount( int( ImGui::GetWindowSize().y / ImGui::GetFontSize() ) );
            resetLabels();
        }

        auto pixRange = actualSize.y - ImGui::GetFontSize();
        if ( onlyTopHalf )
            pixRange *= 2;

        for ( int i = 0; i < labels_.size(); ++i )
        {
            if ( !onlyTopHalf || labels_[i].value <= 0.5f )
            {
                float textW = ImGui::CalcTextSize( labels_[i].text.c_str() ).x;
                drawList->AddText(
                    ImVec2( actualPose.x + style.WindowPadding.x + maxTextSize - textW, actualPose.y + labels_[i].value * pixRange ),
                    ImGui::GetColorU32( SceneColors::get( SceneColors::Labels ).getUInt32() ),
                    labels_[i].text.c_str()
                );
            }
        }
    }

    if ( actualSize.x < maxTextSize + 2 * style.WindowPadding.x + style.FramePadding.x )
        return ImGui::End();

    const std::vector<Color>& colors = texture_.pixels;
    const auto sz = colors.size() >> 1; // only half because remaining colors are all gray
    if ( texture_.filter == FilterType::Discrete )
    {
        auto yStep = actualSize.y / sz;
        if ( onlyTopHalf )
            yStep *= 2;
        for ( int i = 0; i < sz; i++ )
        {
            drawList->AddRectFilled(
                { actualPose.x + style.WindowPadding.x + maxTextSize + style.FramePadding.x,
                actualPose.y + i * yStep },
                { actualPose.x - style.WindowPadding.x + actualSize.x ,
                actualPose.y + ( i + 1 ) * yStep },
                colors[sz - 1 - i].getUInt32() );
        }
    }

    if ( texture_.filter == FilterType::Linear )
    {
        auto yStep = actualSize.y / ( sz - 1 );
        if ( onlyTopHalf )
            yStep *= 2;
        for ( int i = 0; i + 1 < sz; i++ )
        {
            const auto color1 = colors[sz - 1 - i].getUInt32();
            const auto color2 = colors[sz - 2 - i].getUInt32();
            drawList->AddRectFilledMultiColor(
                { actualPose.x + style.WindowPadding.x + maxTextSize + style.FramePadding.x,
                actualPose.y + i * yStep },
                { actualPose.x - style.WindowPadding.x + actualSize.x ,
                actualPose.y + ( i + 1 ) * yStep },
                color1, color1, color2, color2 );
        }
    }

    ImGui::End();
}

Color Palette::getColor( float val )
{
    assert( val >= 0.f && val <= 1.f );

    // only the first row represents the actual palette colours; see `Palette::updateDiscretizatedColors_' for more info
    const std::span<Color> colors( texture_.pixels.data(), texture_.resolution.x );
    if ( val == 1.f )
        return colors.back();

    float dIdx = val * ( colors.size() - 1 );

    if ( texture_.filter == FilterType::Discrete )
        return colors[int( round( dIdx ) )];

    if ( texture_.filter == FilterType::Linear )
    {
        int dId = int( trunc( dIdx ) );
        float c = dIdx - dId;
        return  ( 1 - c ) * colors[dId] + c * colors[dId + 1];
    }

    // unknown FilterType
    return Color();
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

VertUVCoords Palette::getUVcoords( const VertScalars & values, const VertBitSet & region, const VertPredicate & valids ) const
{
    MR_TIMER

    VertUVCoords res;
    res.resizeNoInit( region.size() );
    BitSetParallelFor( region, [&] ( VertId v )
    {
        res[v] = getUVcoord( values[v], contains( valids, v ) );
    } );
    return res;
}

VertUVCoords Palette::getUVcoords( const VertScalars & values, const VertBitSet & region, const VertBitSet * valids ) const
{
    return getUVcoords( values, region, valids ? [valids]( VertId v ) { return valids->test( v ); } : VertPredicate{} );
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
    colors.resize( 2 * sz, Color::gray() );
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

void Palette::updateCustomLabels_()
{
    labels_ = customLabels_;
    for ( auto& label : labels_ )
    {
        label.value = 1.f - getRelativePos( label.value );
    }
    sortLabels_();
}

void Palette::sortLabels_()
{
    std::sort( labels_.begin(), labels_.end(), []( auto itL, auto itR )
    {
        return itL.value < itR.value;
    } );
}

void Palette::resizeCallback_( ImGuiSizeCallbackData* data )
{
    auto palette = ( Palette* )data->UserData;
    if ( !palette )
        return;

    palette->setMaxLabelCount( int( ImGui::GetWindowSize().y / ImGui::GetFontSize() ) );
    palette->resetLabels();
}


std::string Palette::getStringValue( float value )
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

    // although json is a textual format, we open the file in binary mode to get exactly the same result on Windows and Linux
    std::ofstream ofs( path, std::ofstream::binary );
    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer{ builder.newStreamWriter() };
    if ( !ofs || writer->write( root, &ofs ) != 0 )
        return unexpected( "Cannot save preset with name: \"" + name + "\"" );
    ofs.close();

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
