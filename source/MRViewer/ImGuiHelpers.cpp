#include "ImGuiHelpers.h"
#include "imgui_internal.h"
#include "MRMesh/MRBitSet.h"
#include "MRPch/MRSpdlog.h"
#include "MRRibbonButtonDrawer.h"
#include "MRPalette.h"
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include "MRRibbonMenu.h"
#include "MRImGuiImage.h"

namespace ImGui
{

const std::string dragTooltipStr = "Drag with Shift - faster, Alt - slower";

void drawCursorArrow()
{
    auto drawList = ImGui::GetForegroundDrawList();
    auto mousePos = ImGui::GetMousePos();
    mousePos.x += 5.f;

    const auto menuPlugin = MR::getViewerInstance().getMenuPlugin();
    const float scale = menuPlugin ? menuPlugin->menu_scaling() : 1.f;

    const float spaceX = 10 * scale;
    const float sizeX = 12 * scale;
    const float sizeY_2 = 5 * scale;
    // values are calculated so that width of the border line is 1 pixel
    const float shiftLeftX = 2.6f * scale;
    const float shiftRightX = 1.f * scale;
    const float shiftRightY = 1.5f * scale;

    drawList->AddTriangleFilled( ImVec2( mousePos.x - spaceX - sizeX - shiftLeftX, mousePos.y + sizeY_2 ),
                                 ImVec2( mousePos.x - spaceX + shiftRightX, mousePos.y - shiftRightY ),
                                 ImVec2( mousePos.x - spaceX + shiftRightX, mousePos.y + sizeY_2 * 2.f + shiftRightY ), 0xFF000000 );
    drawList->AddTriangleFilled( ImVec2( mousePos.x - spaceX - sizeX, mousePos.y + sizeY_2 ),
                                 ImVec2( mousePos.x - spaceX, mousePos.y ),
                                 ImVec2( mousePos.x - spaceX, mousePos.y + sizeY_2 * 2.f ), 0xFFFFFFFF );

    drawList->AddTriangleFilled( ImVec2( mousePos.x + spaceX - shiftRightX, mousePos.y - shiftRightY ),
                                 ImVec2( mousePos.x + spaceX + sizeX + shiftLeftX, mousePos.y + sizeY_2 ),
                                 ImVec2( mousePos.x + spaceX - shiftRightX, mousePos.y + sizeY_2 * 2.f + shiftRightY ), 0xFF000000 );
    drawList->AddTriangleFilled( ImVec2( mousePos.x + spaceX, mousePos.y ),
                                 ImVec2( mousePos.x + spaceX + sizeX, mousePos.y + sizeY_2 ),
                                 ImVec2( mousePos.x + spaceX, mousePos.y + sizeY_2 * 2.f ), 0xFFFFFFFF );

}

template<typename T>
std::string getRangeStr( T min, T max )
{
    std::string text;
    if ( min > std::numeric_limits<T>::lowest() )
    {
        if ( max < std::numeric_limits<T>::max() )
            text += fmt::format( "valid range [{} - {}]", min, max );
        else
            text += fmt::format( "minimum value {}", min );
    }
    else if ( max < std::numeric_limits<T>::max() )
        text += fmt::format( "maximum value {}", max );
    return text;
}

template<typename T>
void drawTooltip( T min, T max )
{
    static bool inputMode = false;
    if ( IsItemActivated() )
        inputMode = ( GetIO().MouseClicked[0] && GetIO().KeyCtrl ) || GetIO().MouseDoubleClicked[0];

    if ( IsItemActive() )
    {
        if ( !inputMode )
        {
            SetMouseCursor( ImGuiMouseCursor_None );
            drawCursorArrow();
            BeginTooltip();
            Text( "%s", dragTooltipStr.c_str() );
            EndTooltip();
        }

        std::string rangeStr = getRangeStr( min, max );
        if ( !rangeStr.empty() )
        {
            BeginTooltip();
            Text( "%s", rangeStr.c_str() );
            EndTooltip();
        }
    }
}


void PlotCustomHistogram( const char* str_id,
                                 std::function<float( int idx )> values_getter,
                                 std::function<void( int idx )> tooltip,
                                 std::function<void( int idx )> on_click,
                                 int values_count, int values_offset,
                                 float scale_min, float scale_max,
                                 ImVec2 frame_size, int selectedBarId )
{
    if ( frame_size.y < 0.0f )
        return;
    bool encolorSelected = selectedBarId >= 0;

    const ImGuiStyle& style = GetStyle();
    [[maybe_unused]] const ImGuiID id = GetID( str_id );

    if ( frame_size.x == 0.0f )
        frame_size.x = CalcItemWidth();
    if ( frame_size.y == 0.0f )
        frame_size.y = ( style.FramePadding.y * 2 );

    const ImVec2 cursorPos = GetCursorScreenPos();
    ImVec2 max, minPlus, maxPlus;
    max.x = cursorPos.x + frame_size.x; max.y = cursorPos.y + frame_size.y;
    ImVec2 innerMin = cursorPos; innerMin.x += style.FramePadding.x; innerMin.y += style.FramePadding.y;
    ImVec2 innerMax = max; innerMax.x -= style.FramePadding.x; innerMax.y -= style.FramePadding.y;
    if ( ( innerMax.y - innerMin.y ) <= 0.0f )
        return;

    Dummy( frame_size );

    const bool hovered = IsItemHovered();

    // Determine scale from values if not specified
    if ( scale_min == FLT_MAX || scale_max == FLT_MAX )
    {
        float v_min = FLT_MAX;
        float v_max = -FLT_MAX;
        for ( int i = 0; i < values_count; i++ )
        {
            const float v = values_getter( i );
            if ( v != v ) // Ignore NaN values
                continue;
            v_min = std::min( v_min, v );
            v_max = std::max( v_max, v );
        }
        if ( scale_min == FLT_MAX )
            scale_min = v_min;
        if ( scale_max == FLT_MAX )
            scale_max = v_max;
    }

    ImDrawList* drawList  = GetWindowDrawList();
    drawList->AddRectFilled( cursorPos, max, GetColorU32( ImGuiCol_FrameBg ), style.FrameRounding );
    const float border_size = style.FrameBorderSize;
    if ( border_size > 0.0f )
    {
        minPlus.x = cursorPos.x + 1; minPlus.y = cursorPos.y + 1;
        maxPlus.x = max.x + 1; minPlus.y = max.y + 1;
        drawList->AddRect( minPlus, maxPlus, GetColorU32( ImGuiCol_BorderShadow ), style.FrameRounding, ImDrawFlags_RoundCornersAll, border_size );
        drawList->AddRect( cursorPos, max, GetColorU32( ImGuiCol_Border ), style.FrameRounding, ImDrawFlags_RoundCornersAll, border_size );
    }

    const int values_count_min = 1;
    // -1 is not allowed because of marking [0] bar
    int idx_hovered = std::numeric_limits<int>::min();
    if ( values_count >= values_count_min )
    {
        int res_w = std::min( (int) frame_size.x, values_count );
        int item_count = values_count;

        const ImVec2 mousePos = GetIO().MousePos;
        // Tooltip on hover
        if ( hovered && mousePos.x > innerMin.x && mousePos.y > innerMin.y &&mousePos.x < innerMax.x&&mousePos.y < innerMax.y )
        {
            const float t = std::clamp( ( mousePos.x - innerMin.x ) / ( innerMax.x - innerMin.x ), 0.0f, 0.9999f );
            const int v_idx = (int) ( t * item_count );
            IM_ASSERT( v_idx >= 0 && v_idx < values_count );

            tooltip( ( v_idx + values_offset ) % values_count );
            idx_hovered = v_idx;
            if (GetIO().MouseClicked[0])
            {
                on_click((v_idx + values_offset) % values_count);
            }
        }

        const float t_step = 1.0f / (float) res_w;
        const float inv_scale = ( scale_min == scale_max ) ? 0.0f : ( 1.0f / ( scale_max - scale_min ) );

        float v0 = values_getter(( 0 + values_offset ) % values_count );
        float t0 = 0.0f;
        ImVec2 tp0 = ImVec2( t0, 1.0f - std::clamp( ( v0 - scale_min ) * inv_scale, 0.0f, 1.0f ) );                       // Point in the normalized space of our target rectangle
        float histogram_zero_line_t = ( scale_min * scale_max < 0.0f ) ? ( -scale_min * inv_scale ) : ( scale_min < 0.0f ? 0.0f : 1.0f );   // Where does the zero line stands

        const ImU32 col_base = GetColorU32( ImGuiCol_PlotHistogram );
        const ImU32 col_hovered = GetColorU32(ImGuiCol_PlotHistogramHovered);
        ImVec4 col{ 1.0f, 0.2f, 0.2f, 1.0f };
        const ImU32 col_selected = GetColorU32(col);

        for ( int n = 0; n < res_w; n++ )
        {
            const float t1 = t0 + t_step;
            const int v1_idx = (int) ( t0 * item_count + 0.5f );
            IM_ASSERT( v1_idx >= 0 && v1_idx < values_count );
            const float v1 = values_getter( ( v1_idx + values_offset + 1 ) % values_count );
            const ImVec2 tp1 = ImVec2( t1, 1.0f - std::clamp( ( v1 - scale_min ) * inv_scale, 0.0f, 1.0f ) );

            // NB: Draw calls are merged together by the DrawList system. Still, we should render our batch are lower level to save a bit of CPU.
            ImVec2 pos0 = ImVec2( innerMin.x + ( innerMax.x - innerMin.x ) * tp0.x, innerMin.y + ( innerMax.y - innerMin.y ) * tp0.y );
            ImVec2 pos1 = ImVec2( innerMin.x + ( innerMax.x - innerMin.x ) * tp1.x, innerMin.y + ( innerMax.y - innerMin.y ) * histogram_zero_line_t );
            {
                if ( pos1.x >= pos0.x + 2.0f )
                    pos1.x -= 1.0f;
                const int bar_halfthickness = 2;
                auto getBarColor = [&](const int v1_idx)
                {
                    if (abs(v1_idx - idx_hovered) < bar_halfthickness)
                        return col_hovered;
                    if ( encolorSelected && abs(v1_idx - selectedBarId) < bar_halfthickness)
                        return col_selected;
                    return col_base;
                };
                drawList->AddRectFilled( pos0, pos1, getBarColor(v1_idx) );
            }

            t0 = t1;
            tp0 = tp1;
        }
    }
}

bool DragFloatValid( const char* label, float* v, float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags )
{
    bool res = DragFloat( label, v, v_speed, v_min, v_max, format, flags );
    *v = std::clamp( *v, v_min, v_max );
    drawTooltip( v_min, v_max );
    return res;
}

bool DragIntValid( const char *label, int* v, float speed,
                   int v_min, int v_max, const char* format )
{
    auto res = DragInt( label, v, speed, v_min, v_max, format );
    *v = std::clamp( *v, v_min, v_max );
    drawTooltip( v_min, v_max );
    return res;
}

bool InputIntValid( const char* label, int* v, int v_min, int v_max,
    int step, int step_fast, ImGuiInputTextFlags flags )
{
    auto res = InputInt( label, v, step, step_fast, flags );
    *v = std::clamp( *v, v_min, v_max );
    if ( IsItemActive() )
    {
        std::string text = getRangeStr( v_min, v_max );
        if ( !text.empty() )
        {
            BeginTooltip();
            Text( "%s", text.c_str() );
            EndTooltip();
        }
    }
    return res;
}

MultiDragRes DragFloatValid3( const char * label, float* valueArr, float step, float valueMin, float valueMax, const char* format, ImGuiSliderFlags flags, const char* (*tooltips)[3] )
{
    MultiDragRes res;

    ImGuiContext& g = *ImGui::GetCurrentContext();
    ImGuiWindow* window = g.CurrentWindow;
    if (window->SkipItems)
        return res;

    BeginGroup();
    PushID(label);
    constexpr int components = 3;
    PushMultiItemsWidths(components, CalcItemWidth());
    for (int i = 0; i < components; i++)
    {
        PushID(i);
        if (i > 0)
            SameLine(0, g.Style.ItemInnerSpacing.x);
        res.valueChanged = DragFloatValid("", valueArr + i, step, valueMin, valueMax, format, flags) || res.valueChanged;
        if ( tooltips && IsItemHovered() && !IsItemActive() ) 
            SetTooltip( "%s", (*tooltips)[i] );
        res.itemDeactivatedAfterEdit = res.itemDeactivatedAfterEdit || IsItemDeactivatedAfterEdit();
        PopID();
        PopItemWidth();
    }
    PopID();

    const char* label_end = FindRenderedTextEnd(label);
    if (label != label_end)
    {
        SameLine(0, g.Style.ItemInnerSpacing.x);
        TextEx(label, label_end);
    }

    EndGroup();
    return res;
}

bool BeginStatePlugin( const char* label, bool* open, float width )
{
    ImGuiWindow* window = FindWindowByName( label );
    if ( !window )
    {
        float yPos = 0.0f;
        auto menu = MR::getViewerInstance().getMenuPluginAs<MR::RibbonMenu>();
        if ( menu )
            yPos = menu->getTopPanelOpenedHeight() * menu->menu_scaling();
        SetNextWindowPos( ImVec2( GetIO().DisplaySize.x - width, yPos ), ImGuiCond_FirstUseEver );
        SetNextWindowSize( ImVec2( width, 0 ), ImGuiCond_FirstUseEver );
    }
    SetNextWindowSizeConstraints( ImVec2( width, -1.0f ), ImVec2( width, -1.0f ) );
    auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    return Begin( label, open, flags );
}

bool BeginModalNoAnimation( const char* label, bool* open /*= nullptr*/, ImGuiWindowFlags flags /*= 0 */ )
{
    bool started = BeginPopupModal( label, open, flags );
    if ( started )
        GetCurrentContext()->DimBgRatio = 1.0f;
    return started;
}

bool ButtonValid( const char* label, bool valid, const ImVec2& size )
{
    if ( !valid )
    {
        const auto color = GetStyle().Colors[ImGuiCol_TextDisabled];
        PushStyleColor( ImGuiCol_Button, color );
        PushStyleColor( ImGuiCol_ButtonActive, color );
        PushStyleColor( ImGuiCol_ButtonHovered, color );
    }
    bool res = Button( label, size ) && valid;
    if ( !valid )
        PopStyleColor( 3 );

    return res;
}

bool InputIntBitSet( const char* label, int* v, const MR::BitSet& bs, int step /*= 1*/, int step_fast /*= 100*/, ImGuiInputTextFlags flags /*= 0 */ )
{
    int& value = *v;
    const int oldValue = value;

    const int firstValid = int( bs.find_first() );
    const bool validBitSet = firstValid >= 0;
    ImGuiInputTextFlags inputFlag = ImGuiInputTextFlags_None;
    if ( !validBitSet )
    {
        const auto color = GetStyle().Colors[ImGuiCol_TextDisabled];
        PushStyleColor( ImGuiCol_Button, color );
        PushStyleColor( ImGuiCol_ButtonActive, color );
        PushStyleColor( ImGuiCol_ButtonHovered, color );
        inputFlag |= ImGuiInputTextFlags_ReadOnly;
    }

    const bool changed = ImGui::InputInt( label, v, step, step_fast, flags | inputFlag );

    if ( !validBitSet )
    {
        PopStyleColor( 3 );
        value = oldValue;
        return false;
    }

    if ( !bs.test( value ) && ImGui::IsItemDeactivatedAfterEdit() )
    {
        const int lastValid = int( bs.find_last() );
        if ( value < firstValid )
            value = firstValid;
        else if ( value > lastValid )
            value = lastValid;
        else
        {
            if ( changed )
            {
                if ( value > oldValue )
                    value = int( bs.find_next( value ) );
                else
                    while ( !bs.test( --value ) );
            }
            else
                value = int( bs.find_next( value ) );
        }
    }

    return value != oldValue && bs.test( value );
}

bool DragInputInt( const char* label, int* value, float speed /*= 1*/, int min /*= std::numeric_limits<int>::lowest()*/,
                   int max /*= std::numeric_limits<int>::max()*/, const char* format /*= "%d" */, ImGuiSliderFlags flags /*= ImGuiSliderFlags_None*/ )
{
    ImGuiContext& g = *ImGui::GetCurrentContext();
    ImGuiWindow* window = g.CurrentWindow;
    if ( window->SkipItems )
        return false;

    BeginGroup();
    PushID( label );

    int& valueRef = *value;
    const int valueOld = valueRef;

    const std::string labelStr = std::string( "##" ) + label;

    const auto& style = GetStyle();
    const float sizeSide = style.FramePadding.y * 2 + ImGui::GetTextLineHeight();
    SetNextItemWidth( ImMax( 1.0f, CalcItemWidth() - ( sizeSide + style.ItemInnerSpacing.x ) * 2 ) );
    DragInt( labelStr.c_str(), value, speed, min, max, format, flags );
    drawTooltip( min, max );
    ImGui::SameLine( 0, style.ItemInnerSpacing.x );
    ImGui::PushButtonRepeat( true );

    if ( MR::RibbonButtonDrawer::GradientButton( "-", ImVec2( sizeSide, sizeSide ) ) )
        --valueRef;
    ImGui::SameLine( 0, style.ItemInnerSpacing.x );
    if ( MR::RibbonButtonDrawer::GradientButton( "+", ImVec2( sizeSide, sizeSide ) ) )
        ++valueRef;
    ImGui::PopButtonRepeat();
    valueRef = std::clamp( valueRef, min, max );

    PopID();

    const char* label_end = FindRenderedTextEnd( label );
    if ( label != label_end )
    {
        SameLine( 0, g.Style.ItemInnerSpacing.x );
        TextEx( label, label_end );
    }

    EndGroup();

    return valueRef != valueOld;
}

bool Link( const char* label )
{
    auto window = GetCurrentContext()->CurrentWindow;
    assert( window );
    if ( !window )
        return false;

    constexpr unsigned color = MR::Color( 60, 120, 255 ).getUInt32();

    auto linkSize = CalcTextSize( label );

    auto basePos = ImVec2( window->DC.CursorPos.x, 
                           window->DC.CursorPos.y + window->DC.CurrLineTextBaseOffset );
    ImVec2 linkBbMaxPoint( basePos.x + linkSize.x, basePos.y + linkSize.y );
    ImRect linkRect( basePos, linkBbMaxPoint );

    auto linkId = window->GetID( label );
    bool hovered, held;
    bool pressed = ButtonBehavior( linkRect, linkId, &hovered, &held );

    if ( hovered )
    {
        SetMouseCursor( ImGuiMouseCursor_Hand );
        window->DrawList->AddLine( ImVec2( basePos.x, linkBbMaxPoint.y - 1.0f ), 
                                   ImVec2( linkBbMaxPoint.x, linkBbMaxPoint.y - 1.0f ), 
                                   color );
    }

    PushStyleColor( ImGuiCol_Text, color );
    ImGui::Text( "%s", label );
    PopStyleColor();

    return pressed;
}

PaletteChanges Palette(
    const char* label, 
    MR::Palette& palette,
    float width,
    bool* fixZero,
    float speed,
    float min,
    float max,
    const char* format )
{
    using namespace MR;
    int changes = int( PaletteChanges::None );

    const auto& presets = PalettePresets::getPresetNames();
    if ( !presets.empty() )
    {
        ImGui::SetNextItemWidth( width );
        if ( BeginCombo( "Load palette preset", nullptr, ImGuiComboFlags_NoPreview ) )
        {
            for ( int i = 0; i < presets.size(); ++i )
            {
                bool constFalse = false;
                if ( Selectable( presets[i].c_str(), &constFalse ) )
                {
                    PalettePresets::loadPreset( presets[i], palette );
                    if ( fixZero )
                        *fixZero = false;
                    changes = int( PaletteChanges::All );
                    CloseCurrentPopup();
                }
            }
            EndCombo();
        }
    }
    bool fixZeroChanged = false;
    if ( fixZero )
        fixZeroChanged = ImGui::Checkbox( "Set zero to green", fixZero );

    bool isDiscrete = palette.getTexture().filter == MeshTexture::FilterType::Discrete;

    if ( ImGui::Checkbox( "Discrete palette", &isDiscrete ) )
    {
        palette.setFilterType( isDiscrete ? MeshTexture::FilterType::Discrete : MeshTexture::FilterType::Linear );
        changes |= int( PaletteChanges::Texture );
    }

    const auto& params = palette.getParameters();
    if ( isDiscrete )
    {
        ImGui::SameLine();
        int discretization = params.discretization;
        ImGui::SetNextItemWidth( 0.4f * width );
        if ( ImGui::DragIntValid( "Discretization", &discretization, 1, 2, 100 ) )
        {
            palette.setDiscretizationNumber( discretization );
            palette.resetLabels();
            changes |= int( PaletteChanges::Texture );
        }
    }

    assert( params.ranges.size() == 2 || params.ranges.size() == 4 );
    int paletteRangeMode = params.ranges.size() == 2 ? 0 : 1;
    int paletteRangeModeBackUp = paletteRangeMode;
    ImGui::SetNextItemWidth( width );
    ImGui::Combo( "Palette type", &paletteRangeMode, "Even space\0Central zone\0\0" );

    float ranges[4];
    ranges[0] = params.ranges.front();
    ranges[3] = params.ranges.back();
    if ( paletteRangeMode == 1 )
    {
        if ( paletteRangeModeBackUp == 0 )
            ranges[1] = ranges[2] = ( ranges[0] + ranges[3] ) * 0.5f;
        else
        {
            ranges[1] = params.ranges[1];
            ranges[2] = params.ranges[2];
        }
    }

    ImGui::PushItemWidth( 0.8f * width );
    bool rangesChanged = false;
    if ( paletteRangeMode == 0 )
    {
        if ( fixZero && ( *fixZero ) )
        {
            if ( ranges[3] < 0.0f )
                ranges[3] = 0.0f;
            rangesChanged |= ImGui::DragFloatValid( "Min/Max", &ranges[3], speed, 0.0f, max, format );

            if ( rangesChanged || fixZeroChanged )
                ranges[0] = -ranges[3];
        }
        else
        {
            rangesChanged |= ImGui::DragFloatValid( "Max (red)", &ranges[3], speed, min, max, format );
            rangesChanged |= ImGui::DragFloatValid( "Min (blue)", &ranges[0], speed, min, max, format );
        }
    }
    else if ( paletteRangeMode == 1 )
    {
        if ( fixZero && ( *fixZero ) )
        {
            if ( ranges[3] < 0.0f )
                ranges[3] = 0.0f;
            rangesChanged |= ImGui::DragFloatValid( "Max positive / Min negative", &ranges[3], speed, min, max, format );
            if ( rangesChanged || fixZeroChanged )
                ranges[0] = -ranges[3];

            if ( ranges[2] < 0.0f )
                ranges[2] = 0.0f;
            rangesChanged |= ImGui::DragFloatValid( "Min positive / Max negative", &ranges[2], speed, min, max, format );
            if ( rangesChanged || fixZeroChanged )
                ranges[1] = -ranges[2];
        }
        else
        {
            rangesChanged |= ImGui::DragFloatValid( "Max positive (red)", &ranges[3], speed, min, max, format );
            rangesChanged |= ImGui::DragFloatValid( "Min positive (green)", &ranges[2], speed, min, max, format );
            rangesChanged |= ImGui::DragFloatValid( "Max negative (green)", &ranges[1], speed, min, max, format );
            rangesChanged |= ImGui::DragFloatValid( "Min negative (blue)", &ranges[0], speed, min, max, format );
        }
    }
    ImGui::PopItemWidth();

    bool correctOreder = true;
    int orderStep = ( 1 - paletteRangeMode ) * 2 + 1;
    for ( int i = 0; i < 4; )
    {
        int next = i + orderStep;
        if ( next >= 4 )
            break;
        if ( ranges[i] > ranges[next] )
        {
            correctOreder = false;
            break;
        }
        i = next;
    }

    if ( !correctOreder )
    {
        ImGui::PushStyleColor( ImGuiCol_Text, Color::red().getUInt32() );
        ImGui::TextWrapped( "Invalid values order" );
        ImGui::PopStyleColor();
    }
    if ( correctOreder && ( fixZeroChanged || ( paletteRangeMode != paletteRangeModeBackUp ) || rangesChanged ) )
    {
        changes |= int( PaletteChanges::Ranges );
        if ( paletteRangeMode == 0 )
            palette.setRangeMinMax( ranges[0], ranges[3] );
        else
            palette.setRangeMinMaxNegPos( ranges[0], ranges[1], ranges[2], ranges[3] );
    }
    
    std::string popupName = std::string( "Save palette config" ) + std::string( label );
    if ( ImGui::Button( "Save palette as", ImVec2( width, 0 ) ) )
        ImGui::OpenPopup( popupName.c_str() );

    ImVec2 windowSize( 2 * width, 0 );
    ImGui::SetNextWindowPos( ImVec2( ( ImGui::GetIO().DisplaySize.x - windowSize.x ) / 2.f, ( ImGui::GetIO().DisplaySize.y - windowSize.y ) / 2.f ), ImGuiCond_Always );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
    if ( !ImGui::BeginModalNoAnimation( popupName.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar ) )
        return PaletteChanges( changes );

    static std::string curentPaletteName;
    ImGui::InputText( "Config name", curentPaletteName );

    const float btnWidth = 80.0f * ImGui::GetIO().DisplayFramebufferScale.x;
    if ( ImGui::ButtonValid( "Save", !curentPaletteName.empty(), ImVec2( btnWidth, 0 ) ) )
    {
        std::error_code ec;
        if ( std::filesystem::is_regular_file( PalettePresets::getPalettePresetsFolder() / ( curentPaletteName + ".json" ), ec ) )
        {
            OpenPopup( "Palette already exists##PaletteHelper" );
        }
        else
        {
            PalettePresets::savePreset( curentPaletteName, palette );
            ImGui::CloseCurrentPopup();
        }
    }

    bool closeTopPopup = false;
    if ( ImGui::BeginModalNoAnimation( "Palette already exists##PaletteHelper", nullptr,
                                 ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar ) )
    {
        ImGui::Text( "Palette preset with this name already exists, override?" );
        auto w = GetContentRegionAvail().x;
        auto p = GetStyle().FramePadding.x;
        if ( ImGui::Button( "Yes", ImVec2( ( w - p ) * 0.5f, 0 ) ) )
        {
            PalettePresets::savePreset( curentPaletteName, palette );
            closeTopPopup = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine( 0, p );
        if ( ImGui::Button( "No", ImVec2( ( w - p ) * 0.5f, 0 ) ) )
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    if ( closeTopPopup )
        ImGui::CloseCurrentPopup();

    ImGui::SameLine();
    const auto& style = ImGui::GetStyle();
    ImGui::SetCursorPosX( windowSize.x - btnWidth - style.WindowPadding.x );
    if ( ImGui::Button( "Cancel", ImVec2( btnWidth, 0 ) ) )
        ImGui::CloseCurrentPopup();

    ImGui::EndPopup();

    return PaletteChanges( changes );
}

void Image( const MR::ImGuiImage& image, const ImVec2& size, const MR::Color& multColor )
{
    MR::Vector4 tintColor = ( MR::Vector4f ) multColor;
    Image( image, size, ImVec4( tintColor.x, tintColor.y, tintColor.z, tintColor.w ) );
}

void Image( const MR::ImGuiImage& image, const ImVec2& size, const ImVec4& multColor )
{
    Image( image.getImTextureId(), size, ImVec2( 0, 1 ), ImVec2( 1, 0 ), multColor );
}

MR::Vector2i GetImagePointerCoord( const MR::ImGuiImage& image, const ImVec2& size, const ImVec2& imagePos )
{
    const auto& io = ImGui::GetIO();
    return  { int( ( io.MousePos.x - imagePos.x ) / size.x * image.getImageWidth() ), int( ( size.y - io.MousePos.y + imagePos.y ) / size.y * image.getImageHeight() ) };
}

} // namespace ImGui
