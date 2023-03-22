#include "MRRibbonButtonDrawer.h"
#include "MRRibbonMenu.h"
#include "MRColorTheme.h"
#include "MRRibbonConstants.h"
#include "MRImGuiImage.h"
#include "MRShortcutManager.h"
#include "ImGuiHelpers.h"
#include "MRRibbonIcons.h"
#include "imgui_internal.h"

namespace MR
{

std::vector<std::unique_ptr<MR::ImGuiImage>> RibbonButtonDrawer::textures_ = std::vector<std::unique_ptr<MR::ImGuiImage>>( int( RibbonButtonDrawer::TextureType::Count ) );

void DrawCustomArrow( ImDrawList* drawList, const ImVec2& startPoint, const ImVec2& midPoint, const ImVec2& endPoint, ImU32 col, float thickness )
{
    drawList->PathLineTo( startPoint );
    drawList->PathLineTo( midPoint );
    drawList->PathLineTo( endPoint );
    drawList->PathStroke( col, 0, thickness );

    const float radius = thickness * 0.5f;
    drawList->AddCircleFilled( startPoint, radius, col );
    drawList->AddCircleFilled( midPoint, radius, col );
    drawList->AddCircleFilled( endPoint, radius, col );
}

void RibbonButtonDrawer::InitGradientTexture()
{
    auto& textureM = GetTexture( TextureType::Mono );
    if ( !textureM )
        textureM = std::make_unique<ImGuiImage>();
    MeshTexture data;
    data.resolution = Vector2i( 1, 1 );
    data.pixels = { Color::white() };
    data.filter = FilterType::Linear;
    textureM->update( data );

    auto& textureG = GetTexture( TextureType::Gradient );
    if ( !textureG )
        textureG = std::make_unique<ImGuiImage>();
    data.resolution = Vector2i( 1, 2 );
    data.pixels = {
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradientStart ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradientEnd )
    };
    data.filter = FilterType::Linear;
    textureG->update( data );


    auto& textureR = GetTexture( TextureType::RainbowRect );
    if ( !textureR )
        textureR = std::make_unique<ImGuiImage>();
    const int resX = 4;
    const int resY = 2;
    data.resolution = Vector2i( resX, resY );
    data.pixels.resize( resX * resY );
    float h, r, g, b;
    for ( int i = 0; i < resX; ++i )
    {
        h = ( 3.5f - 2.f * i / ( resX - 1.f ) ) / 6.f;
        ImGui::ColorConvertHSVtoRGB( h, 1.f, 1.f, r, g, b );
        data.pixels[i] = Color( r, g, b );

        h = ( 5.f + 2.f * i / ( resX - 1.f ) ) / 6.f;
        if ( h > 1.f ) h -= 1.f;
        ImGui::ColorConvertHSVtoRGB( h, 1.f, 1.f, r, g, b );
        data.pixels[i + resX] = Color( r, g, b );
    }
    data.filter = FilterType::Linear;
    textureR->update( data );

}

std::unique_ptr<MR::ImGuiImage>& RibbonButtonDrawer::GetTexture( TextureType type )
{
    const int typeInt = int( type );
    assert( typeInt < textures_.size() && typeInt >= 0 );
    return textures_[typeInt];
}

bool RibbonButtonDrawer::GradientButton( const char* label, const ImVec2& size /*= ImVec2( 0, 0 ) */, ImGuiKey key )
{
    auto& texture = GetTexture( TextureType::Gradient );
    auto checkKey = [] ( ImGuiKey passedKey )
    {
        if ( passedKey == ImGuiKey_None )
            return false;
        if ( passedKey == ImGuiKey_Enter || passedKey == ImGuiKey_KeypadEnter )
            return ImGui::IsKeyPressed( ImGuiKey_Enter ) || ImGui::IsKeyPressed( ImGuiKey_KeypadEnter );
        return ImGui::IsKeyPressed( passedKey );
    };
    if ( !texture )
        return ImGui::Button( label, size ) || checkKey( key );

    ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_Text, ImVec4( 1, 1, 1, 1 ) );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 labelSize = ImGui::CalcTextSize( label, NULL, true );

    int pushedStyleNum = 1;
	ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );

	auto framePadding = style.FramePadding;
    if ( size.y == 0 )
    {
        framePadding.y = cGradientButtonFramePadding;
        if ( auto menu = getViewerInstance().getMenuPlugin() )
            framePadding.y *= menu->menu_scaling();
    }
    else if ( size.y > 0 )
	{
        framePadding.y = ( size.y - ImGui::CalcTextSize( label ).y ) / 2.f;
	}
    if ( size.x > 0 )
	{
		framePadding.x = ( size.x - ImGui::CalcTextSize( label ).x ) / 2.f;
    }
	++pushedStyleNum;
	ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, framePadding );

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 realSize = ImGui::CalcItemSize( size, labelSize.x + style.FramePadding.x * 2.0f, labelSize.y + style.FramePadding.y * 2.0f );
    const ImRect bb( pos, ImVec2( pos.x + realSize.x, pos.y + realSize.y ) );

    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded(
        texture->getImTextureId(),
        bb.Min, bb.Max,
        ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
        Color::white().getUInt32(), style.FrameRounding );

    auto res = ImGui::Button( label, size ) || checkKey( key );

    ImGui::PopStyleVar( pushedStyleNum );
    ImGui::PopStyleColor( 2 );
    return res;
}

bool RibbonButtonDrawer::GradientButtonCommonSize( const char* label, const ImVec2& size, ImGuiKey key )
{
    return GradientButton( label, ImVec2( size.x, size.y == 0.0f ? ImGui::GetFrameHeight() : size.y ), key );
}

bool RibbonButtonDrawer::GradientButtonValid( const char* label, bool valid, const ImVec2& size /* = ImVec2(0, 0) */ )
{
    auto& texture = GetTexture( TextureType::Gradient );
    if ( !texture )
        return ImGui::ButtonValid( label, valid, size );

    ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_Text, ImVec4( 1, 1, 1, 1 ) );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 labelSize = ImGui::CalcTextSize( label, NULL, true );

    int pushedStyleNum = 1;
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    if ( size.y == 0 )
    {
        auto framePadding = style.FramePadding;
        framePadding.y = cGradientButtonFramePadding;
        if ( auto menu = getViewerInstance().getMenuPlugin() )
            framePadding.y *= menu->menu_scaling();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, framePadding );
        ++pushedStyleNum;
    }

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 realSize = ImGui::CalcItemSize( size, labelSize.x + style.FramePadding.x * 2.0f, labelSize.y + style.FramePadding.y * 2.0f );
    const ImRect bb( pos, ImVec2( pos.x + realSize.x, pos.y + realSize.y ) );

    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded(
        texture->getImTextureId(),
        bb.Min, bb.Max,
        ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
        Color::white().getUInt32(), style.FrameRounding );

    auto res = ImGui::ButtonValid( label, valid, size );

    ImGui::PopStyleVar( pushedStyleNum );
    ImGui::PopStyleColor( 2 );
    return res;
}

bool RibbonButtonDrawer::GradientCheckbox( const char* label, bool* value )
{
    auto& texture = GetTexture( TextureType::Gradient );
    if ( !texture )
        return  ImGui::Checkbox( label, value );

    const auto bgColor = ImGui::GetColorU32( ImGuiCol_FrameBg );

    ImGui::PushStyleColor( ImGuiCol_FrameBg, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_CheckMark, ImVec4( 1, 1, 1, 1 ) );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 1.5f );

    const auto menu = getViewerInstance().getMenuPlugin();
    const float scaling = menu ? menu->menu_scaling() : 1.0f;
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { cCheckboxPadding * scaling, cCheckboxPadding * scaling } );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    const ImGuiStyle& style = ImGui::GetStyle();
    const float clickSize = ImGui::GetFrameHeight();

    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb( pos, ImVec2( pos.x + clickSize, pos.y + clickSize ) );

    if ( value && *value )
        ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded(
            texture->getImTextureId(),
            bb.Min, bb.Max,
            ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
            Color::white().getUInt32(), style.FrameRounding );

    //code of this lambda is copied from ImGui::Checkbox in order to decrease thickness and change appearance of the check mark
    auto drawCustomCheckbox = [bgColor] ( const char* label, bool* v )
    {
        if ( !ImGui::GetCurrentContext() || !v)
            return false;

        ImGuiContext& g = *ImGui::GetCurrentContext();
        ImGuiWindow* window = g.CurrentWindow;
        if ( !window || window->SkipItems )
            return false;

        const ImGuiStyle& style = ImGui::GetStyle();
        const ImGuiID id = window->GetID( label );
        const ImVec2 label_size = ImGui::CalcTextSize( label, NULL, true );

        const float square_sz = ImGui::GetFrameHeight();
        const ImVec2 pos = window->DC.CursorPos;
        const ImRect total_bb( pos, ImVec2( pos.x + square_sz + ( label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f ), pos.y + label_size.y + style.FramePadding.y * 2.0f ) );
        ImGui::ItemSize( total_bb, style.FramePadding.y );
        if ( !ImGui::ItemAdd( total_bb, id ) )
        {
            IMGUI_TEST_ENGINE_ITEM_INFO( id, label, g.LastItemData.StatusFlags | ImGuiItemStatusFlags_Checkable | ( *v ? ImGuiItemStatusFlags_Checked : 0 ) );
            return false;
        }

        bool hovered, held;
        bool pressed = ImGui::ButtonBehavior( total_bb, id, &hovered, &held );
        if ( pressed )
        {
            *v = !( *v );
            ImGui::MarkItemEdited( id );
        }

        const ImRect check_bb( pos, ImVec2( pos.x + square_sz, pos.y + square_sz ) );
        ImGui::RenderNavHighlight( total_bb, id );

        if ( *v )
            ImGui::RenderFrame( check_bb.Min, check_bb.Max, ImGui::GetColorU32( ( held && hovered ) ? ImGuiCol_FrameBgActive : hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg ), true, style.FrameRounding * 0.5f );
        else
            ImGui::RenderFrame( check_bb.Min, check_bb.Max, ImGui::GetColorU32( ( held && hovered ) ? ImGuiCol_FrameBgActive : hovered ? ImGuiCol_FrameBgHovered : bgColor ), true, style.FrameRounding * 0.5f );

        ImU32 check_col = ImGui::GetColorU32( ImGuiCol_CheckMark );
        bool mixed_value = ( g.LastItemData.InFlags & ImGuiItemFlags_MixedValue ) != 0;
        if ( mixed_value )
        {
            // Undocumented tristate/mixed/indeterminate checkbox (#2644)
            // This may seem awkwardly designed because the aim is to make ImGuiItemFlags_MixedValue supported by all widgets (not just checkbox)
            ImVec2 pad( ImMax( 1.0f, IM_FLOOR( square_sz / 3.6f ) ), ImMax( 1.0f, IM_FLOOR( square_sz / 3.6f ) ) );
            window->DrawList->AddRectFilled( { check_bb.Min.x + pad.x,  check_bb.Min.y + pad.y }, { check_bb.Max.x - pad.x, check_bb.Max.y - pad.y }, check_col, style.FrameRounding );
        }
        else if ( *v )
        {
            const float pad = ImMax( 1.0f, IM_FLOOR( square_sz / 6.0f ) );
            auto renderCustomCheckmark = [] ( ImDrawList* draw_list, ImVec2 pos, ImU32 col, float sz )
            {
                const float thickness = ImMax( sz * 0.15f, 1.0f );
                sz -= thickness * 0.5f;
                pos = ImVec2( pos.x + thickness * 0.25f, pos.y + thickness * 0.25f );

                const float half = sz * 0.5f;
                const float ninth = sz / 9.0f;
                const ImVec2 startPoint { pos.x + ninth, pos.y + half };
                const ImVec2 anglePoint { pos.x + half, pos.y + sz - ninth };
                const ImVec2 endPoint { pos.x + sz - ninth, pos.y + ninth * 2.0f };

                draw_list->PathLineTo( startPoint );
                draw_list->PathLineTo( anglePoint );
                draw_list->PathLineTo( endPoint );
                draw_list->PathStroke( col, 0, thickness );

                const float radius = thickness * 0.5f;
                draw_list->AddCircleFilled( startPoint, radius, col );
                draw_list->AddCircleFilled( anglePoint, radius, col );
                draw_list->AddCircleFilled( endPoint, radius, col );
            };
            renderCustomCheckmark( window->DrawList, { check_bb.Min.x +  pad, check_bb.Min.y + pad }, check_col, square_sz - pad * 2.0f );
        }

        ImVec2 label_pos = ImVec2( check_bb.Max.x + style.ItemInnerSpacing.x, check_bb.Min.y + style.FramePadding.y );
        if ( g.LogEnabled )
            ImGui::LogRenderedText( &label_pos, mixed_value ? "[~]" : *v ? "[x]" : "[ ]" );
        if ( label_size.x > 0.0f )
            ImGui::RenderText( label_pos, label );

        IMGUI_TEST_ENGINE_ITEM_INFO( id, label, g.LastItemData.StatusFlags | ImGuiItemStatusFlags_Checkable | ( *v ? ImGuiItemStatusFlags_Checked : 0 ) );
        return pressed;
    };

    auto res = drawCustomCheckbox( label, value );

    ImGui::PopStyleVar( 2 );
    ImGui::PopStyleColor( 2 );
    return res;
}

bool RibbonButtonDrawer::GradientCheckboxItem( const MenuItemInfo& item, bool* value ) const
{
    bool res = GradientCheckbox( ( "##" + item.item->name() ).c_str(), value );
    const float spacing = ImGui::GetStyle().ItemInnerSpacing.x + 3;
    ImGui::SameLine( 0.f, spacing );
    const float height = ImGui::GetTextLineHeight();
    drawButtonIcon( item, DrawButtonParams{.itemSize = ImVec2( height + 4, height + 4 ), .iconSize = height / scaling_,
                                           .rootType = DrawButtonParams::RootType::Toolbar } );
    ImGui::SameLine( 0.f, spacing );
    std::string name = item.caption.empty() ? item.item->name() : item.caption;
    ImGui::Text( "%s", name.c_str());
    return res;
}

bool RibbonButtonDrawer::GradientCheckboxMixed( const char* label, bool* value, bool mixed )
{
    if ( mixed )
    {
        ImGuiContext& g = *ImGui::GetCurrentContext();
        ImGuiItemFlags backup_item_flags = g.CurrentItemFlags;
        g.CurrentItemFlags |= ImGuiItemFlags_MixedValue;
        const bool changed = GradientCheckbox( label, value );
        g.CurrentItemFlags = backup_item_flags;
        return changed;
    }
    else
    {
        return GradientCheckbox( label, value );
    }
}

bool RibbonButtonDrawer::GradientRadioButton( const char* label, int* value, int v_button )
{
    auto& texture = GetTexture( TextureType::Gradient );
    if ( !texture )
        return ImGui::RadioButton( label, value, v_button );

    const auto bgColor = ImGui::GetColorU32( ImGuiCol_FrameBg );

    ImGui::PushStyleColor( ImGuiCol_FrameBg, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_CheckMark, ImVec4( 1, 1, 1, 1 ) );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 1.0f );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    const auto menu = getViewerInstance().getMenuPlugin();
    const float scaling = menu ? menu->menu_scaling() : 1.0f;

    const float clickSize = cRadioButtonSize * scaling;

    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb( pos, ImVec2( pos.x + clickSize, pos.y + clickSize ) );

    if ( value && *value == v_button )
        ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded(
            texture->getImTextureId(),
            bb.Min, bb.Max,
            ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
            Color::white().getUInt32(), clickSize * 0.5f );

    //code of this lambda is copied from ImGui::RadioBitton in order to decrease size of the central circle
    auto drawCustomRadioButton = [bgColor, scaling, clickSize]( const char* label, int* v, int v_button )
    {     
        if ( !ImGui::GetCurrentContext() || !v )
            return false;

        ImGuiContext& g = *ImGui::GetCurrentContext();
        ImGuiWindow* window = g.CurrentWindow;
        if ( !window || window->SkipItems )
            return false;

        const ImGuiStyle& style = ImGui::GetStyle();
        const ImGuiID id = window->GetID( label );
        const ImVec2 label_size = ImGui::CalcTextSize( label, NULL, true );

        const auto menu = getViewerInstance().getMenuPlugin();
        const ImVec2 pos = window->DC.CursorPos;
        const ImRect check_bb( pos, ImVec2( pos.x + clickSize, pos.y + clickSize ) );
        const ImRect total_bb( pos, ImVec2( pos.x + clickSize + ( label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f ), pos.y + label_size.y + style.FramePadding.y * 2.0f ) );
        ImGui::ItemSize( total_bb, style.FramePadding.y );
        if ( !ImGui::ItemAdd( total_bb, id ) )
            return false;

        ImVec2 center = check_bb.GetCenter();
        const float radius = clickSize * 0.5f;

        bool hovered, held;
        bool pressed = ImGui::ButtonBehavior( total_bb, id, &hovered, &held );
        if ( pressed )
        {
            ImGui::MarkItemEdited( id );
            *v = v_button;
        }

        ImGui::RenderNavHighlight( total_bb, id );

        const bool active = *v == v_button;
        if ( active )
        {
            window->DrawList->AddCircleFilled( center, radius, ImGui::GetColorU32( ( held && hovered ) ? ImGuiCol_FrameBgActive : hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg ), 16 );
            const float pad = ImMax( 1.0f, IM_FLOOR( clickSize * 0.3f ) );
            window->DrawList->AddCircleFilled( center, radius - pad, ImGui::GetColorU32( ImGuiCol_CheckMark ), 16 );
        }
        else
        {
            window->DrawList->AddCircleFilled( center, radius, ImGui::GetColorU32( ( held && hovered ) ? ImGuiCol_FrameBgActive : hovered ? ImGuiCol_FrameBgHovered : bgColor ), 16 );
            if ( style.FrameBorderSize > 0.0f )
            {
                const float thickness = 1.5f * scaling;
                window->DrawList->AddCircle( center, radius, ImGui::GetColorU32( ImGuiCol_Border ), 16, style.FrameBorderSize * thickness );
            }
        }

        ImVec2 label_pos = ImVec2( check_bb.Max.x + style.ItemInnerSpacing.x, check_bb.Min.y + style.FramePadding.y );
        ImGui::RenderText( label_pos, label );

        IMGUI_TEST_ENGINE_ITEM_INFO( id, label, g.LastItemData.StatusFlags );
        return pressed;
    };

    auto res = drawCustomRadioButton( label, value, v_button );

    ImGui::PopStyleVar();
    ImGui::PopStyleColor( 2 );
    return res;
}

/// copy of internal ImGui method
void ColorEditRestoreHS( const float* col, float* H, float* S, float* V )
{
    ImGuiContext& g = *ImGui::GetCurrentContext();
    if ( g.ColorEditLastColor != ImGui::ColorConvertFloat4ToU32( ImVec4( col[0], col[1], col[2], 0 ) ) )
        return;

    if ( *S == 0.0f || ( *H == 0.0f && g.ColorEditLastHue == 1 ) )
        *H = g.ColorEditLastHue;

    if ( *V == 0.0f )
        *S = g.ColorEditLastSat;
}

bool RibbonButtonDrawer::GradientColorEdit4( const char* label, Vector4f& color, ImGuiColorEditFlags flags /*= ImGuiColorEditFlags_None */ )
{
    const auto& style = ImGui::GetStyle();
    ImVec2 framePadding( 8.f, 3.f );
    float colorEditFrameRounding( 2.f );
    ImVec2 itemInnerSpacing( 12, style.ItemInnerSpacing.y );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, framePadding );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, colorEditFrameRounding );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, itemInnerSpacing );

    /// Copy of ImGui code. Required to implement own code
    using namespace ImGui;
    float* col = &color.x;

    ImGuiContext& g = *GetCurrentContext();
    ImGuiWindow* window = g.CurrentWindow;
    if ( window->SkipItems )
        return false;

    const float square_sz = GetFrameHeight();
    const float w_full = CalcItemWidth();
    const float w_button = ( flags & ImGuiColorEditFlags_NoSmallPreview ) ? 0.0f : ( square_sz * 1.5f + style.ItemInnerSpacing.x );
    const float w_inputs = w_full - w_button;
    const char* label_display_end = FindRenderedTextEnd( label );
    g.NextItemData.ClearFlags();

    BeginGroup();
    PushID( label );

    // If we're not showing any slider there's no point in doing any HSV conversions
    const ImGuiColorEditFlags flags_untouched = flags;
    if ( flags & ImGuiColorEditFlags_NoInputs )
        flags = ( flags & ( ~ImGuiColorEditFlags_DisplayMask_ ) ) | ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_NoOptions;

    // Context menu: display and modify options (before defaults are applied)
    if ( !( flags & ImGuiColorEditFlags_NoOptions ) )
        ColorEditOptionsPopup( col, flags );

    // Read stored options
    if ( !( flags & ImGuiColorEditFlags_DisplayMask_ ) )
        flags |= ( g.ColorEditOptions & ImGuiColorEditFlags_DisplayMask_ );
    if ( !( flags & ImGuiColorEditFlags_DataTypeMask_ ) )
        flags |= ( g.ColorEditOptions & ImGuiColorEditFlags_DataTypeMask_ );
    if ( !( flags & ImGuiColorEditFlags_PickerMask_ ) )
        flags |= ( g.ColorEditOptions & ImGuiColorEditFlags_PickerMask_ );
    if ( !( flags & ImGuiColorEditFlags_InputMask_ ) )
        flags |= ( g.ColorEditOptions & ImGuiColorEditFlags_InputMask_ );
    flags |= ( g.ColorEditOptions & ~( ImGuiColorEditFlags_DisplayMask_ | ImGuiColorEditFlags_DataTypeMask_ | ImGuiColorEditFlags_PickerMask_ | ImGuiColorEditFlags_InputMask_ ) );
    IM_ASSERT( ImIsPowerOfTwo( flags & ImGuiColorEditFlags_DisplayMask_ ) ); // Check that only 1 is selected
    IM_ASSERT( ImIsPowerOfTwo( flags & ImGuiColorEditFlags_InputMask_ ) );   // Check that only 1 is selected

    const bool alpha = ( flags & ImGuiColorEditFlags_NoAlpha ) == 0;
    const bool hdr = ( flags & ImGuiColorEditFlags_HDR ) != 0;
    const int components = alpha ? 4 : 3;

    // Convert to the formats we need
    float f[4] = { col[0], col[1], col[2], alpha ? col[3] : 1.0f };
    if ( ( flags & ImGuiColorEditFlags_InputHSV ) && ( flags & ImGuiColorEditFlags_DisplayRGB ) )
        ColorConvertHSVtoRGB( f[0], f[1], f[2], f[0], f[1], f[2] );
    else if ( ( flags & ImGuiColorEditFlags_InputRGB ) && ( flags & ImGuiColorEditFlags_DisplayHSV ) )
    {
        // Hue is lost when converting from greyscale rgb (saturation=0). Restore it.
        ColorConvertRGBtoHSV( f[0], f[1], f[2], f[0], f[1], f[2] );
        ColorEditRestoreHS( col, &f[0], &f[1], &f[2] );
    }
    int i[4] = { IM_F32_TO_INT8_UNBOUND( f[0] ), IM_F32_TO_INT8_UNBOUND( f[1] ), IM_F32_TO_INT8_UNBOUND( f[2] ), IM_F32_TO_INT8_UNBOUND( f[3] ) };

    bool value_changed = false;
    bool value_changed_as_float = false;

    const ImVec2 pos = window->DC.CursorPos;
    const float inputs_offset_x = ( style.ColorButtonPosition == ImGuiDir_Left ) ? w_button : 0.0f;
    window->DC.CursorPos.x = pos.x + inputs_offset_x;

    if ( ( flags & ( ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_DisplayHSV ) ) != 0 && ( flags & ImGuiColorEditFlags_NoInputs ) == 0 )
    {
        // RGB/HSV 0..255 Sliders
        const float w_item_one = ImMax( 1.0f, IM_FLOOR( ( w_inputs - ( style.ItemInnerSpacing.x ) * ( components - 1 ) ) / ( float )components ) );
        const float w_item_last = ImMax( 1.0f, IM_FLOOR( w_inputs - ( w_item_one + style.ItemInnerSpacing.x ) * ( components - 1 ) ) );

        const bool hide_prefix = ( w_item_one <= CalcTextSize( ( flags & ImGuiColorEditFlags_Float ) ? "M:0.000" : "M:000" ).x );
        static const char* ids[4] = { "##X", "##Y", "##Z", "##W" };
        static const char* fmt_table_int[3][4] =
        {
            {   "%3d",   "%3d",   "%3d",   "%3d" }, // Short display
            { "R:%3d", "G:%3d", "B:%3d", "A:%3d" }, // Long display for RGBA
            { "H:%3d", "S:%3d", "V:%3d", "A:%3d" }  // Long display for HSVA
        };
        static const char* fmt_table_float[3][4] =
        {
            {   "%0.3f",   "%0.3f",   "%0.3f",   "%0.3f" }, // Short display
            { "R:%0.3f", "G:%0.3f", "B:%0.3f", "A:%0.3f" }, // Long display for RGBA
            { "H:%0.3f", "S:%0.3f", "V:%0.3f", "A:%0.3f" }  // Long display for HSVA
        };
        const int fmt_idx = hide_prefix ? 0 : ( flags & ImGuiColorEditFlags_DisplayHSV ) ? 2 : 1;

        for ( int n = 0; n < components; n++ )
        {
            if ( n > 0 )
                SameLine( 0, style.ItemInnerSpacing.x );
            SetNextItemWidth( ( n + 1 < components ) ? w_item_one : w_item_last );

            // FIXME: When ImGuiColorEditFlags_HDR flag is passed HS values snap in weird ways when SV values go below 0.
            if ( flags & ImGuiColorEditFlags_Float )
            {
                value_changed |= DragFloat( ids[n], &f[n], 1.0f / 255.0f, 0.0f, hdr ? 0.0f : 1.0f, fmt_table_float[fmt_idx][n] );
                value_changed_as_float |= value_changed;
            }
            else
            {
                value_changed |= DragInt( ids[n], &i[n], 1.0f, 0, hdr ? 0 : 255, fmt_table_int[fmt_idx][n] );
            }
            if ( !( flags & ImGuiColorEditFlags_NoOptions ) )
                OpenPopupOnItemClick( "context", ImGuiPopupFlags_MouseButtonRight );
        }
    }
    else if ( ( flags & ImGuiColorEditFlags_DisplayHex ) != 0 && ( flags & ImGuiColorEditFlags_NoInputs ) == 0 )
    {
        // RGB Hexadecimal Input
        char buf[64];
        if ( alpha )
            ImFormatString( buf, IM_ARRAYSIZE( buf ), "#%02X%02X%02X%02X", ImClamp( i[0], 0, 255 ), ImClamp( i[1], 0, 255 ), ImClamp( i[2], 0, 255 ), ImClamp( i[3], 0, 255 ) );
        else
            ImFormatString( buf, IM_ARRAYSIZE( buf ), "#%02X%02X%02X", ImClamp( i[0], 0, 255 ), ImClamp( i[1], 0, 255 ), ImClamp( i[2], 0, 255 ) );
        SetNextItemWidth( w_inputs );
        if ( InputText( "##Text", buf, IM_ARRAYSIZE( buf ), ImGuiInputTextFlags_CharsHexadecimal | ImGuiInputTextFlags_CharsUppercase ) )
        {
            value_changed = true;
            char* p = buf;
            while ( *p == '#' || ImCharIsBlankA( *p ) )
                p++;
            i[0] = i[1] = i[2] = 0;
            i[3] = 0xFF; // alpha default to 255 is not parsed by scanf (e.g. inputting #FFFFFF omitting alpha)
            int r;
            if ( alpha )
                r = sscanf( p, "%02X%02X%02X%02X", ( unsigned int* )&i[0], ( unsigned int* )&i[1], ( unsigned int* )&i[2], ( unsigned int* )&i[3] ); // Treat at unsigned (%X is unsigned)
            else
                r = sscanf( p, "%02X%02X%02X", ( unsigned int* )&i[0], ( unsigned int* )&i[1], ( unsigned int* )&i[2] );
            IM_UNUSED( r ); // Fixes C6031: Return value ignored: 'sscanf'.
        }
        if ( !( flags & ImGuiColorEditFlags_NoOptions ) )
            OpenPopupOnItemClick( "context", ImGuiPopupFlags_MouseButtonRight );
    }

    ImGuiWindow* picker_active_window = NULL;
    if ( !( flags & ImGuiColorEditFlags_NoSmallPreview ) )
    {
        const float button_offset_x = ( ( flags & ImGuiColorEditFlags_NoInputs ) || ( style.ColorButtonPosition == ImGuiDir_Left ) ) ? 0.0f : w_inputs + style.ItemInnerSpacing.x;
        window->DC.CursorPos = ImVec2( pos.x + button_offset_x, pos.y );

        const ImVec4 col_v4( col[0], col[1], col[2], alpha ? col[3] : 1.0f );

        const float frameH = GetFrameHeight();
        float off = 0.f;
        ImRect bb( window->DC.CursorPos, ImVec2( window->DC.CursorPos.x + frameH * 1.5f, window->DC.CursorPos.y + frameH ) );
        if ( !( flags & ImGuiColorEditFlags_NoBorder ) )
        {
            off = 2.f;
            Vector3f hsv, hsvBg;
            ImGui::ColorConvertRGBtoHSV( col[0], col[1], col[2], hsv[0], hsv[1], hsv[2] );
            const Vector4f bgColor = Vector4f( ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Background ) );
            ImGui::ColorConvertRGBtoHSV( bgColor[0], bgColor[1], bgColor[2], hsvBg[0], hsvBg[1], hsvBg[2] );
            const bool isRainbow = std::fabs( hsv[2] - hsvBg[2] ) < 0.5 && ( hsv[1] < 0.5f || hsv[2] < 0.5f );
            const Color imageColor = isRainbow ? Color::white() : ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Borders );
            auto& texture = GetTexture( isRainbow ? TextureType::RainbowRect : TextureType::Mono );
            ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded( texture->getImTextureId(),
                bb.Min, bb.Max, ImVec2( 0.f, 0.f ), ImVec2( 1.f, 1.f ), imageColor.getUInt32(), style.FrameRounding );
            bb.Expand( -off );
        }

        const ImVec2 btnSize = bb.GetSize();
        window->DC.CursorPos.x += off;
        window->DC.CursorPos.y += off;
        if ( ColorButton( "##ColorButton", col_v4, flags | ImGuiColorEditFlags_NoBorder, btnSize ) )
        {
            if ( !( flags & ImGuiColorEditFlags_NoPicker ) )
            {
                // Store current color and open a picker
                g.ColorPickerRef = col_v4;
                OpenPopup( "picker" );
                ImVec2 winPos = g.LastItemData.Rect.GetBL();
                winPos.y += style.ItemSpacing.y;
                SetNextWindowPos( winPos );
            }
        }
        window->DC.CursorPos.x += off;
        window->DC.CursorPos.y -= off;
        if ( !( flags & ImGuiColorEditFlags_NoOptions ) )
            OpenPopupOnItemClick( "context", ImGuiPopupFlags_MouseButtonRight );

        if ( BeginPopup( "picker" ) )
        {
            if ( g.CurrentWindow->BeginCount == 1 )
            {
                picker_active_window = g.CurrentWindow;
                if ( label != label_display_end )
                {
                    TextEx( label, label_display_end );
                    Spacing();
                }
                ImGuiColorEditFlags picker_flags_to_forward = ImGuiColorEditFlags_DataTypeMask_ | ImGuiColorEditFlags_PickerMask_ | ImGuiColorEditFlags_InputMask_ | ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_AlphaBar;
                ImGuiColorEditFlags picker_flags = ( flags_untouched & picker_flags_to_forward ) | ImGuiColorEditFlags_DisplayMask_ | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaPreviewHalf;
                SetNextItemWidth( square_sz * 12.0f ); // Use 256 + bar sizes?
                value_changed |= ColorPicker4( "##picker", col, picker_flags, &g.ColorPickerRef.x );
            }
            EndPopup();
        }
    }

    if ( label != label_display_end && !( flags & ImGuiColorEditFlags_NoLabel ) )
    {
        // Position not necessarily next to last submitted button (e.g. if style.ColorButtonPosition == ImGuiDir_Left),
        // but we need to use SameLine() to setup baseline correctly. Might want to refactor SameLine() to simplify this.
        SameLine( 0.0f, style.ItemInnerSpacing.x );
        window->DC.CursorPos.x = pos.x + ( ( flags & ImGuiColorEditFlags_NoInputs ) ? w_button : w_full + style.ItemInnerSpacing.x );
        TextEx( label, label_display_end );
    }

    // Convert back
    if ( value_changed && picker_active_window == NULL )
    {
        if ( !value_changed_as_float )
            for ( int n = 0; n < 4; n++ )
                f[n] = i[n] / 255.0f;
        if ( ( flags & ImGuiColorEditFlags_DisplayHSV ) && ( flags & ImGuiColorEditFlags_InputRGB ) )
        {
            g.ColorEditLastHue = f[0];
            g.ColorEditLastSat = f[1];
            ColorConvertHSVtoRGB( f[0], f[1], f[2], f[0], f[1], f[2] );
            g.ColorEditLastColor = ColorConvertFloat4ToU32( ImVec4( f[0], f[1], f[2], 0 ) );
        }
        if ( ( flags & ImGuiColorEditFlags_DisplayRGB ) && ( flags & ImGuiColorEditFlags_InputHSV ) )
            ColorConvertRGBtoHSV( f[0], f[1], f[2], f[0], f[1], f[2] );

        col[0] = f[0];
        col[1] = f[1];
        col[2] = f[2];
        if ( alpha )
            col[3] = f[3];
    }

    PopID();
    EndGroup();

    // Drag and Drop Target
    // NB: The flag test is merely an optional micro-optimization, BeginDragDropTarget() does the same test.
    if ( ( g.LastItemData.StatusFlags & ImGuiItemStatusFlags_HoveredRect ) && !( flags & ImGuiColorEditFlags_NoDragDrop ) && BeginDragDropTarget() )
    {
        bool accepted_drag_drop = false;
        if ( const ImGuiPayload* payload = AcceptDragDropPayload( IMGUI_PAYLOAD_TYPE_COLOR_3F ) )
        {
            memcpy( ( float* )col, payload->Data, sizeof( float ) * 3 ); // Preserve alpha if any //-V512 //-V1086
            value_changed = accepted_drag_drop = true;
        }
        if ( const ImGuiPayload* payload = AcceptDragDropPayload( IMGUI_PAYLOAD_TYPE_COLOR_4F ) )
        {
            memcpy( ( float* )col, payload->Data, sizeof( float ) * components );
            value_changed = accepted_drag_drop = true;
        }

        // Drag-drop payloads are always RGB
        if ( accepted_drag_drop && ( flags & ImGuiColorEditFlags_InputHSV ) )
            ColorConvertRGBtoHSV( col[0], col[1], col[2], col[0], col[1], col[2] );
        EndDragDropTarget();
    }

    // When picker is being actively used, use its active id so IsItemActive() will function on ColorEdit4().
    if ( picker_active_window && g.ActiveId != 0 && g.ActiveIdWindow == picker_active_window )
        g.LastItemData.ID = g.ActiveId;

    if ( value_changed && g.LastItemData.ID != 0 ) // In case of ID collision, the second EndGroup() won't catch g.ActiveId
        MarkItemEdited( g.LastItemData.ID );

    ImGui::PopStyleVar( 3 );
    return value_changed;
}

bool RibbonButtonDrawer::CustomCombo( const char* label, int* v, const std::vector<std::string>& options, bool showPreview, const std::vector<std::string>& tooltips, const std::string& defaultText )
{
    assert( tooltips.empty() || tooltips.size() == options.size() );

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, MR::StyleConsts::CustomCombo::framePadding );

    auto context = ImGui::GetCurrentContext();
    ImGuiWindow* window = context->CurrentWindow;
    const auto& style = ImGui::GetStyle();
    const ImVec2 pos = window->DC.CursorPos;
    const float arrowSize = 2 * style.FramePadding.y + ImGui::GetTextLineHeight();
    if ( !showPreview )
        ImGui::PushItemWidth( arrowSize + style.FramePadding.x * 0.5f );

    float itemWidth = ( context->NextItemData.Flags & ImGuiNextItemDataFlags_HasWidth ) ? context->NextItemData.Width : window->DC.ItemWidth;
    const ImRect boundingBox( pos, { pos.x + itemWidth, pos.y + arrowSize } );
    const ImRect arrowBox( { pos.x + boundingBox.GetWidth() - boundingBox.GetHeight() * 6.0f / 7.0f, pos.y }, boundingBox.Max );

    auto res = ImGui::BeginCombo( label, nullptr, ImGuiComboFlags_NoArrowButton);
    if ( showPreview )
    {
        const char* previewText = ( v && *v >= 0 ) ? options[*v].data() : defaultText.data();
        ImGui::RenderTextClipped( { boundingBox.Min.x + style.FramePadding.x, boundingBox.Min.y + style.FramePadding.y }, { boundingBox.Max.x - arrowSize, boundingBox.Max.y }, previewText, nullptr, nullptr );
    }

    const float halfHeight = arrowBox.GetHeight() * 0.5f;
    const float arrowHeight = arrowBox.GetHeight() * 5.0f / 42.0f;
    const float arrowWidth = arrowBox.GetWidth() * 2.0f / 15.0f;

    const float thickness = ImMax( arrowBox.GetHeight() * 0.075f, 1.0f );
    
    const ImVec2 arrowPos{ arrowBox.Min.x, arrowBox.Min.y - thickness };
    const ImVec2 startPoint{ arrowPos.x + arrowWidth, arrowPos.y + halfHeight };
    const ImVec2 midPoint{ arrowPos.x + 2 * arrowWidth, arrowPos.y + halfHeight + arrowHeight };
    const ImVec2 endPoint{ arrowPos.x + 3 * arrowWidth, arrowPos.y + halfHeight };

    DrawCustomArrow( window->DrawList, startPoint, midPoint, endPoint, ImGui::GetColorU32( ImGuiCol_Text ), thickness );

    ImGui::PopStyleVar();

    if ( !res )
        return false;
    
    for ( int i = 0; i < int(options.size()); ++i )
    {
        ImGui::PushID( (label + std::to_string( i )).c_str() );
        if ( ImGui::Selectable( options[i].c_str(), *v == i ) )
            *v = i;
        
        if ( !tooltips.empty() )
            ImGui::SetTooltipIfHovered( tooltips[i], Viewer::instanceRef().getMenuPlugin()->menu_scaling() );

        ImGui::PopID();        
    }

    ImGui::EndCombo();
    if ( !showPreview )
        ImGui::PopItemWidth();
    return true;
}

bool RibbonButtonDrawer::CustomCollapsingHeader( const char* label, ImGuiTreeNodeFlags flags, int issueCount )
{
    const auto& style = ImGui::GetStyle();
    auto pos = ImGui::GetCursorScreenPos();
    pos.x += style.FramePadding.x;
    pos.y += style.FramePadding.y;

    auto context = ImGui::GetCurrentContext();
    auto window = context->CurrentWindow;
    auto drawList = window->DrawList;

    const float height = ImGui::GetTextLineHeight();
    const float width = ImGui::GetTextLineHeight();
    const float textWidth = ImGui::CalcTextSize( label ).x;

    auto res = ImGui::CollapsingHeader( label, flags );
    for ( int i = 0; i < issueCount; ++i )
    {
        drawList->AddCircleFilled( { pos.x + textWidth + 3.0f * width + i * width, pos.y + height / 2.0f }, height / 3.0f, Color( 0.886f, 0.267f, 0.267f, 1.0f ).getUInt32() );
    }

    const auto isActive = ImGui::IsItemActive();
    const auto isHovered = ImGui::IsItemHovered( ImGuiHoveredFlags_AllowWhenBlockedByActiveItem );

    const auto windowBgColor = ImGui::GetStyleColorVec4( ImGuiCol_WindowBg );
    const auto headerColor = ImGui::GetStyleColorVec4( ( isActive && isHovered ) ? ImGuiCol_HeaderActive : isHovered ? ImGuiCol_HeaderHovered : ImGuiCol_Header );
    const float alpha = headerColor.w;

    const ImVec4 blendedHeaderColor
    {
        windowBgColor.x + ( headerColor.x - windowBgColor.x ) * alpha,
        windowBgColor.y + ( headerColor.y - windowBgColor.y ) * alpha,
        windowBgColor.z + ( headerColor.z - windowBgColor.z ) * alpha,
        1.0f
    };
    

    drawList->AddRectFilled( pos, { pos.x + width, pos.y + height }, ImGui::GetColorU32( blendedHeaderColor ) );

    const float thickness = ImMax( height * 0.15f, 1.0f );
    if ( res )
    {
        const auto halfWidth = width * 0.5f;
        const auto horIndent = height * 0.25f;
        const auto vertIndent = height * 7.5f / 20.0f;

        const ImVec2 startPoint { pos.x + horIndent, pos.y + vertIndent };
        const ImVec2 midPoint{ pos.x + halfWidth, pos.y + height - vertIndent };
        const ImVec2 endPoint{ pos.x + width - horIndent, pos.y + vertIndent };

        DrawCustomArrow( drawList, startPoint, midPoint, endPoint, ImGui::GetColorU32( ImGuiCol_Text ), thickness );
    }
    else
    {
        const auto halfHeight = height * 0.5f;
        const auto horIndent = width * 7.5f / 20.0f;
        const auto vertIndent = height * 0.25f;

        const ImVec2 startPoint{ pos.x + horIndent, pos.y + vertIndent };
        const ImVec2 midPoint{ pos.x + width - horIndent, pos.y + halfHeight };
        const ImVec2 endPoint{ pos.x + horIndent, pos.y + height - vertIndent };

        DrawCustomArrow( drawList, startPoint, midPoint, endPoint, ImGui::GetColorU32( ImGuiCol_Text ), thickness );
    }

    return res;
}

RibbonButtonDrawer::ButtonItemWidth RibbonButtonDrawer::calcItemWidth( const MenuItemInfo& item, DrawButtonParams::SizeType sizeType ) const
{
    ButtonItemWidth res;
    if ( sizeType == DrawButtonParams::SizeType::Big )
    {
        const float cMinItemSize = cRibbonItemMinWidth * scaling_;

        float maxTextWidth = 0.f;
        for ( const auto& i : item.captionSize.splitInfo )
            maxTextWidth = std::max( maxTextWidth, i.second );

        res.baseWidth = maxTextWidth + 2 * cRibbonButtonWindowPaddingX * scaling_;

        if ( item.item->type() == RibbonItemType::ButtonWithDrop )
        {
            auto additionalSize = 3 * cSmallIconSize * scaling_;
            if ( cMinItemSize - res.baseWidth < additionalSize )
                res.baseWidth += additionalSize;
        }

        if ( res.baseWidth < cMinItemSize )
            res.baseWidth = cMinItemSize;
        return res;
    }
    else if ( sizeType == DrawButtonParams::SizeType::SmallText )
    {
        res.baseWidth = ( cSmallIconSize +
                          2.0f * cRibbonButtonWindowPaddingX +
                          2.0f * cRibbonItemInterval ) * scaling_ +
            item.captionSize.baseSize;

        if ( item.item->type() == RibbonItemType::ButtonWithDrop )
            res.additionalWidth = cSmallItemDropSizeModifier * ( cSmallIconSize + 2.0f * cRibbonButtonWindowPaddingX ) * scaling_;
    }
    else
    {
        res.baseWidth = ( cSmallIconSize + 2.0f * cRibbonButtonWindowPaddingX ) * scaling_;

        if ( item.item->type() == RibbonItemType::ButtonWithDrop )
            res.additionalWidth = cSmallItemDropSizeModifier * res.baseWidth;
    }
    return res;
}

void RibbonButtonDrawer::drawButtonItem( const MenuItemInfo& item, const DrawButtonParams& params ) const
{
    std::string requirements = getRequirements_( item.item );

    bool dropItem = item.item->type() == RibbonItemType::ButtonWithDrop;

    ImVec2 itemSize = params.itemSize;
    if ( dropItem && params.sizeType == DrawButtonParams::SizeType::Small )
        itemSize.x += params.itemSize.x * cSmallItemDropSizeModifier;

    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0, 0 ) );
    ImGui::BeginChild( ( "##childGroup" + item.item->name() ).c_str(), itemSize, false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse );
    ImGui::PopStyleVar();

    ImGui::BeginGroup();

    int colorChanged = pushRibbonButtonColors_( requirements.empty(), item.item->isActive(), params.rootType );
    bool pressed = ImGui::ButtonEx( ( "##wholeChildBtn" + item.item->name() ).c_str(), itemSize, ImGuiButtonFlags_AllowItemOverlap );
    ImGui::SetItemAllowOverlap();

    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    float fontScale = 1.f;
    if ( font ) {
        if ( params.iconSize != 0 )
            font->Scale = params.iconSize / cBigIconSize;
        else if ( params.sizeType != DrawButtonParams::SizeType::Big )
            font->Scale = cSmallIconSize / cBigIconSize;
        fontScale = font->Scale;
        ImGui::PushFont( font );
    }

    auto imageRequiredSize = std::round( 32.0f * fontScale * scaling_ );
    ImVec2 iconRealSize = ImVec2( imageRequiredSize, imageRequiredSize );
    bool needWhiteIcon = !requirements.empty() || item.item->isActive() || params.rootType != DrawButtonParams::Ribbon;
    auto* imageIcon = RibbonIcons::findByName( item.item->name(), iconRealSize.x, needWhiteIcon ?
                                               RibbonIcons::ColorType::White : RibbonIcons::ColorType::Colored,
                                               RibbonIcons::IconType::RibbonItemIcon );

    if ( !imageIcon )
        iconRealSize = ImGui::CalcTextSize( item.icon.c_str() );

    if ( params.sizeType != DrawButtonParams::SizeType::SmallText )
    {
        ImGui::SetCursorPosX( ( params.itemSize.x - iconRealSize.x ) * 0.5f );
    }
    else
    {
        if ( !imageIcon )
            ImGui::SetCursorPosX( ImGui::GetStyle().WindowPadding.x );
        else
            ImGui::SetCursorPosX( ImGui::GetStyle().WindowPadding.x * 0.5f );
    }

    if ( params.sizeType != DrawButtonParams::SizeType::Big )
        ImGui::SetCursorPosY( ( params.itemSize.y - iconRealSize.y ) * 0.5f );
    else
    {
        if ( !imageIcon )
            ImGui::SetCursorPosY( 2.0f * ImGui::GetStyle().WindowPadding.y );
        else
            ImGui::SetCursorPosY( ImGui::GetStyle().WindowPadding.y );
    }

    if ( !imageIcon )
        ImGui::Text( "%s", item.icon.c_str() );
    else
    {
        ImVec4 multColor = ImVec4( 1, 1, 1, 1 );
        if ( needWhiteIcon )
            multColor = ImGui::GetStyleColorVec4( ImGuiCol_Text );
        ImGui::Image( *imageIcon, iconRealSize, multColor );
    }

    if ( font )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    if ( params.sizeType == DrawButtonParams::SizeType::Big )
    {
        const float availableHeight = ImGui::GetContentRegionAvail().y;
        const int numLines = int( item.captionSize.splitInfo.size() );
        const float textHeight = numLines * ImGui::GetTextLineHeight() + ( numLines - 1 ) * ImGui::GetStyle().ItemSpacing.y;

        if ( !imageIcon )
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ( availableHeight - textHeight ) * 0.5f );
        else
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ( availableHeight - textHeight ) * 0.5f - ImGui::GetStyle().WindowPadding.y );

        for ( const auto& i : item.captionSize.splitInfo )
        {
            ImGui::SetCursorPosX( ( params.itemSize.x - i.second ) * 0.5f );
            ImGui::TextUnformatted( &i.first.front(), &i.first.back() + 1 );
        }
    }
    else if ( params.sizeType == DrawButtonParams::SizeType::SmallText )
    {
        ImGui::SameLine();
        ImGui::SetCursorPosY( ( params.itemSize.y - ImGui::GetTextLineHeight() ) * 0.5f );
        const auto& caption = item.caption.empty() ? item.item->name() : item.caption;
        ImGui::Text( "%s", caption.c_str() );
    }

    if ( colorChanged > 0 )
        ImGui::PopStyleColor( colorChanged );

    ImGui::EndGroup();

    if ( pressed )
        onPressAction_( item.item, requirements.empty() );

    if ( ImGui::IsItemHovered() )
        drawTooltip_( item, requirements );

    if ( dropItem )
        drawButtonDropItem_( item, params );
    ImGui::EndChild();
    ImGui::PopStyleVar();
}

void RibbonButtonDrawer::drawButtonIcon( const MenuItemInfo& item, const DrawButtonParams& params ) const
{
    ImGui::BeginGroup();

    int colorChanged = pushRibbonButtonColors_( true, false, params.rootType );

    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    float fontScale = 1.f;
    if ( font )
    {
        if ( params.iconSize != 0 )
            font->Scale = params.iconSize / cBigIconSize;
        else if ( params.sizeType != DrawButtonParams::SizeType::Big )
            font->Scale = cSmallIconSize / cBigIconSize;
        fontScale = font->Scale;
        ImGui::PushFont( font );
    }

    auto imageRequiredSize = std::round( 32.0f * fontScale * scaling_ );
    ImVec2 iconRealSize = ImVec2( imageRequiredSize, imageRequiredSize );
    bool needWhiteIcon = params.rootType != DrawButtonParams::Ribbon;
    auto* imageIcon = RibbonIcons::findByName( item.item->name(), iconRealSize.x, needWhiteIcon ?
                                               RibbonIcons::ColorType::White : RibbonIcons::ColorType::Colored,
                                               RibbonIcons::IconType::RibbonItemIcon );

    if ( !imageIcon )
        iconRealSize = ImGui::CalcTextSize( item.icon.c_str() );

    ImVec2 cursorPos = ImGui::GetCursorPos();
    cursorPos.x += ( params.itemSize.x - iconRealSize.x ) / 2.f;
    cursorPos.y += ( params.itemSize.y - iconRealSize.y ) / 2.f;
    ImGui::SetCursorPos( cursorPos );

    if ( !imageIcon )
        ImGui::Text( "%s", item.icon.c_str() );
    else
    {
        ImVec4 multColor = ImVec4( 1, 1, 1, 1 );
        if ( needWhiteIcon )
            multColor = ImGui::GetStyleColorVec4( ImGuiCol_Text );
        ImGui::Image( *imageIcon, iconRealSize, multColor );
    }

    if ( font )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    if ( colorChanged > 0 )
        ImGui::PopStyleColor( colorChanged );

    ImGui::EndGroup();
}

bool RibbonButtonDrawer::drawCustomStyledButton( const char* icon, const ImVec2& size, float iconSize )
{
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding );
    ImGui::PushStyleColor( ImGuiCol_Button, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrab ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );

    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    if ( font )
    {
        font->Scale = iconSize / RibbonFontManager::getFontSizeByType( RibbonFontManager::FontType::Icons );
        ImGui::PushFont( font );
    }

    bool pressed = ImGui::Button( icon, size );

    if ( font )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    ImGui::PopStyleColor( 3 );
    ImGui::PopStyleVar();
    return pressed;
}

void RibbonButtonDrawer::drawButtonDropItem_( const MenuItemInfo& item, const DrawButtonParams& params ) const
{
    float iconSize = params.iconSize * 0.5f;
    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    if ( font )
    {
        const float fontSize = RibbonFontManager::getFontSizeByType( RibbonFontManager::FontType::Icons );
        if ( params.sizeType == DrawButtonParams::SizeType::Big )
            font->Scale = iconSize / fontSize;
        else
            font->Scale = iconSize * 1.5f / fontSize;
        ImGui::PushFont( font );
    }
    ImVec2 itemSize = ImVec2( ImGui::GetFrameHeight(), ImGui::GetFrameHeight() );
    ImVec2 dropBtnPos;
    if ( params.sizeType == DrawButtonParams::SizeType::Small )
    {
        itemSize.x = params.itemSize.x * cSmallItemDropSizeModifier;
        itemSize.y = params.itemSize.y;
        dropBtnPos.x = params.itemSize.x;
        dropBtnPos.y = 0.0f;
    }
    else if ( params.sizeType == DrawButtonParams::SizeType::SmallText )
    {
        itemSize.x = params.itemSize.y;
        itemSize.y = params.itemSize.y;
        dropBtnPos.x = params.itemSize.x - itemSize.x;
        dropBtnPos.y = 0.0f;
    }
    else
    {
        assert( params.sizeType == DrawButtonParams::SizeType::Big );
        dropBtnPos.x = params.itemSize.x - itemSize.x;
        dropBtnPos.y = params.itemSize.y - itemSize.y;
    }
    ImGui::SetCursorPos( dropBtnPos );
    auto absMinPos = ImGui::GetCurrentContext()->CurrentWindow->DC.CursorPos;

    auto name = "##DropDown" + item.item->name();
    auto nameWindow = name + "Popup";
    bool menuOpened = ImGui::IsPopupOpen( nameWindow.c_str() );

    bool dropBtnEnabled = !item.item->dropItems().empty();

    int pushedColors = pushRibbonButtonColors_( dropBtnEnabled, menuOpened, params.rootType );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding );
    bool comboPressed = ImGui::Button( name.c_str(), itemSize ) && dropBtnEnabled;

    auto iconRealSize = ImGui::CalcTextSize( "\xef\x81\xb8" ); //down icon
    ImGui::SetCursorPosX( dropBtnPos.x + ( itemSize.x - iconRealSize.x + 1 ) * 0.5f );
    ImGui::SetCursorPosY( dropBtnPos.y + ( itemSize.y - iconRealSize.y - 1 ) * 0.5f );
    ImGui::Text( "%s", "\xef\x81\xb8" );
    
    ImGui::PopStyleVar();
    ImGui::PopStyleColor( pushedColors );

    if ( font )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    // manage menu popup
    if ( comboPressed && !menuOpened )
        ImGui::OpenPopup( nameWindow.c_str() );

    if ( !menuOpened )
        return;

    if ( ImGuiWindow* menuWindow = ImGui::FindWindowByName( nameWindow.c_str() ) )
        if ( menuWindow->WasActive )
        {
            ImRect frame;
            frame.Min = absMinPos;
            frame.Max = ImVec2( frame.Min.x + ImGui::GetFrameHeight(), frame.Min.y + ImGui::GetFrameHeight() );
            ImVec2 expectedSize = ImGui::CalcWindowNextAutoFitSize( menuWindow );
            menuWindow->AutoPosLastDirection = ImGuiDir_Down;
            ImRect rectOuter = ImGui::GetPopupAllowedExtentRect( menuWindow );
            ImVec2 pos = ImGui::FindBestWindowPosForPopupEx( frame.GetBL(), expectedSize, &menuWindow->AutoPosLastDirection, rectOuter, frame, ImGuiPopupPositionPolicy_ComboBox );
            ImGui::SetNextWindowPos( pos );
        }

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_Popup | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove;
    ImGui::Begin( nameWindow.c_str(), NULL, window_flags );
    if ( menuOpened )
    {
        drawDropList_( item.item );
        ImGui::EndPopup();
    }
}

void RibbonButtonDrawer::drawDropList_( const std::shared_ptr<RibbonMenuItem>& baseDropItem ) const
{
    auto dropList = baseDropItem->dropItems();
    if ( dropList.empty() )
        ImGui::CloseCurrentPopup();
    for ( int i = 0; i < dropList.size(); ++i )
    {
        const auto& dropItem = dropList[i];
        bool itemWithDrop = dropItem->type() == RibbonItemType::ButtonWithDrop;

        if ( i > 0 && itemWithDrop )
            ImGui::Separator();

        std::string requirements = getRequirements_( dropItem );

        std::string caption = dropItem->name();

        const auto& schema = RibbonSchemaHolder::schema();
        auto it = schema.items.find( dropItem->name() );
        if ( it == schema.items.end() )
        {
            auto pressed = ImGui::MenuItem( ( caption + "##dropItem" ).c_str(), nullptr, dropItem->isActive(), requirements.empty() );
            if ( pressed )
                onPressAction_( dropItem, requirements.empty() );            
        }
        else
        {
            const auto& item = it->second;

            if ( !item.caption.empty() )
                caption = item.caption;

            const auto ySize = ( cSmallIconSize + 2 * cRibbonButtonWindowPaddingY ) * scaling_;
            const auto width = calcItemWidth( item, DrawButtonParams::SizeType::SmallText );

            DrawButtonParams params;
            params.sizeType = DrawButtonParams::SizeType::SmallText;
            params.iconSize = cSmallIconSize;
            params.itemSize.y = ySize;
            params.itemSize.x = width.baseWidth + width.additionalWidth + 2.0f * cRibbonButtonWindowPaddingX * scaling_;
            drawButtonItem( item, params );
        }

        if ( itemWithDrop )
        {
            if ( ImGui::BeginMenu( ( "More...##recursiveDropMenu" + dropItem->name() ).c_str(), requirements.empty() ) )
            {
                drawDropList_( dropItem );
                ImGui::EndMenu();
            }
            if ( i + 1 < dropList.size() )
                ImGui::Separator();
        }
    }
}

void RibbonButtonDrawer::drawTooltip_( const MenuItemInfo& item, const std::string& requirements ) const
{
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 0, 0 ) );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( cRibbonButtonWindowPaddingX * scaling_, cRibbonButtonWindowPaddingY * scaling_ ) );
    std::string tooltip = item.item->getDynamicTooltip();
    if ( tooltip.empty() )
        tooltip = item.tooltip;

    const auto& caption = item.caption.empty() ? item.item->name() : item.caption;

    std::string fullText;
    fullText = caption;
    std::string shortcutStr;

    if ( shortcutManager_ )
    {
        auto shortcut = shortcutManager_->findShortcutByName( item.item->name() );
        if ( shortcut )
        {
            shortcutStr = " (" + ShortcutManager::getKeyFullString( *shortcut ) + ")";
            fullText += shortcutStr;
        }
    }

    if ( !tooltip.empty() )
    {
        fullText += '\n';
        fullText += tooltip;
    }
    if ( !requirements.empty() )
    {
        fullText += '\n';
        fullText += requirements;
    }
    auto textSize = ImGui::CalcTextSize( fullText.c_str(), NULL, false, 400.f );

    ImGui::SetNextWindowContentSize( textSize );
    ImGui::BeginTooltip();
    ImGui::Text( "%s%s", caption.c_str(), shortcutStr.c_str() );
    if ( !tooltip.empty() )
    {
        ImGui::PushStyleColor( ImGuiCol_Text, ImGui::GetStyleColorVec4( ImGuiCol_TextDisabled ) );
        ImGui::TextWrapped( "%s", tooltip.c_str() );
        ImGui::PopStyleColor();
    }
    if ( !requirements.empty() )
    {
        ImGui::PushStyleColor( ImGuiCol_Text, Color::red().getUInt32() );
        ImGui::TextWrapped( "%s", requirements.c_str() );
        ImGui::PopStyleColor();
    }
    ImGui::EndTooltip();
    ImGui::PopStyleVar( 2 );
}


int RibbonButtonDrawer::pushRibbonButtonColors_( bool enabled, bool active, DrawButtonParams::RootType rootType ) const
{
    if ( active )
    {
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActiveHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActiveClicked ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_Button, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonActive ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TextEnabled ).getUInt32() );
        return 4;
    }
    if ( !enabled )
    {
        if ( rootType == DrawButtonParams::RootType::Header )
        {
            auto tabTextColor = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText );
            tabTextColor.a = 77;
            ImGui::PushStyleColor( ImGuiCol_Text, tabTextColor.getUInt32() );
        }
        else
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TextDisabled ).getUInt32() );
    }
    else
    {
        if ( rootType == DrawButtonParams::RootType::Header )
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText ).getUInt32() );
        else
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Text ).getUInt32() );
    }

    ImGui::PushStyleColor( ImGuiCol_Button, Color( 0, 0, 0, 0 ).getUInt32() );
    if ( rootType == DrawButtonParams::RootType::Ribbon )
    {
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::RibbonButtonClicked ).getUInt32() );
    }
    else if ( rootType == DrawButtonParams::RootType::Toolbar )
    {
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarClicked ).getUInt32() );
    }
    else 
    {
        assert( rootType == DrawButtonParams::RootType::Header );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabClicked ).getUInt32() );
    }
    return 4;
}

}
