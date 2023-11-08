#include "MRUIStyle.h"
#include "MRImGuiImage.h"
#include "MRRibbonButtonDrawer.h"
#include "MRColorTheme.h"
#include "MRRibbonConstants.h"
#include "MRViewerInstance.h"
#include "MRRibbonFontManager.h"
#include "MRViewer.h"
#include "ImGuiHelpers.h"
#include "ImGuiMenu.h"
#include "imgui_internal.h"
#include "MRMesh/MRVector4.h"


ImVec2 operator+( const ImVec2& a, const ImVec2& b )
{
    return ImVec2( a.x + b.x, a.y + b.y );
}
ImVec2 operator-( const ImVec2& a, const ImVec2& b )
{
    return ImVec2( a.x - b.x, a.y - b.y );
}

namespace MR
{


namespace UI
{

enum class TextureType
{
    Mono,
    Gradient,
    GradientBtn,
    RainbowRect,
    Count
};

std::vector<std::unique_ptr<MR::ImGuiImage>> textures = std::vector<std::unique_ptr<MR::ImGuiImage>>( int( TextureType::Count ) );

std::unique_ptr<MR::ImGuiImage>& getTexture( TextureType type )
{
    const int typeInt = int( type );
    assert( typeInt < textures.size() && typeInt >= 0 );
    return textures[typeInt];
}

//////////////////////////////////////////////////////////////////////////

class StyleParamHolder
{
public:
    ~StyleParamHolder()
    {
        ImGui::PopStyleVar( varCount );
        ImGui::PopStyleColor( colorCount );
    }

    void addVar( ImGuiStyleVar var, float val )
    {
        ImGui::PushStyleVar( var, val );
        ++varCount;
    }
    void addVar( ImGuiStyleVar var, const ImVec2& val )
    {
        ImGui::PushStyleVar( var, val );
        ++varCount;
    }
    void addColor( ImGuiCol colorName, const Color& val )
    {
        ImGui::PushStyleColor( colorName, val.getUInt32() );
        ++colorCount;
    }

private:
    int varCount{ 0 };
    int colorCount{ 0 };
};

//////////////////////////////////////////////////////////////////////////

bool checkKey( ImGuiKey passedKey )
{
    if ( passedKey == ImGuiKey_None )
        return false;
    bool pressed = false;
    if ( passedKey == ImGuiKey_Enter || passedKey == ImGuiKey_KeypadEnter )
        pressed =  ImGui::IsKeyPressed( ImGuiKey_Enter ) || ImGui::IsKeyPressed( ImGuiKey_KeypadEnter );
    else 
        pressed = ImGui::IsKeyPressed( passedKey );
    return pressed && ImGui::GetIO().KeyMods == ImGuiMod_None;
}

//////////////////////////////////////////////////////////////////////////


void init()
{
    auto& textureM = getTexture( TextureType::Mono );
    if ( !textureM )
        textureM = std::make_unique<ImGuiImage>();
    MeshTexture data;
    data.resolution = Vector2i( 1, 1 );
    data.pixels = { Color::white() };
    data.filter = FilterType::Linear;
    textureM->update( data );


    auto& textureG = getTexture( TextureType::Gradient );
    if ( !textureG )
        textureG = std::make_unique<ImGuiImage>();
    data.resolution = Vector2i( 1, 2 );
    data.pixels = {
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradientStart ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradientEnd )
    };
    data.filter = FilterType::Linear;
    textureG->update( data );


    auto& textureGb = getTexture( TextureType::GradientBtn );
    if ( !textureGb )
        textureGb = std::make_unique<ImGuiImage>();
    data.resolution = Vector2i( 4, 2 );
    data.pixels = {
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnStart ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnHoverStart ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnActiveStart ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnDisableStart ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnEnd ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnHoverEnd ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnActiveEnd ),
        ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnDisableEnd ),
    };
    data.filter = FilterType::Linear;
    textureGb->update( data );


    auto& textureR = getTexture( TextureType::RainbowRect );
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

bool buttonEx( const char* label, bool active, const Vector2f& size_arg /*= Vector2f( 0, 0 )*/, ImGuiButtonFlags flags /*= ImGuiButtonFlags_None*/ )
{
    // copy from ImGui::ButtonEx and replaced visualize part
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if ( window->SkipItems )
        return false;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImGuiID id = window->GetID( label );
    const ImVec2 label_size = ImGui::CalcTextSize( label, NULL, true );

    ImVec2 pos = window->DC.CursorPos;
    if ( ( flags & ImGuiButtonFlags_AlignTextBaseLine ) && style.FramePadding.y < window->DC.CurrLineTextBaseOffset ) // Try to vertically align buttons that are smaller/have no padding so that text baseline matches (bit hacky, since it shouldn't be a flag)
        pos.y += window->DC.CurrLineTextBaseOffset - style.FramePadding.y;
    ImVec2 size = ImGui::CalcItemSize( ImVec2(size_arg), label_size.x + style.FramePadding.x * 2.0f, label_size.y + style.FramePadding.y * 2.0f );

    const ImRect bb( pos, pos + size );
    ImGui::ItemSize( size, style.FramePadding.y );
    if ( !ImGui::ItemAdd( bb, id ) )
        return false;

    if ( g.LastItemData.InFlags & ImGuiItemFlags_ButtonRepeat )
        flags |= ImGuiButtonFlags_Repeat;

    bool hovered, held;
    bool pressed = ImGui::ButtonBehavior( bb, id, &hovered, &held, flags );

    // Render
    ImGui::RenderNavHighlight( bb, id );

    // replaced part
    auto& texture = getTexture( TextureType::GradientBtn );
    if ( texture )
    {
        const float textureU = 0.125f + ( !active ? 0.75f : ( held && hovered ) ? 0.5f : hovered ? 0.25f : 0.f );
        window->DrawList->AddImageRounded(
            texture->getImTextureId(),
            bb.Min, bb.Max,
            ImVec2( textureU, 0.25f ), ImVec2( textureU, 0.75f ),
            Color::white().getUInt32(), style.FrameRounding );
    }
    else
    {
        const ImGuiCol colIdx = ( !active ? ImGuiCol_TextDisabled : ( held && hovered ) ? ImGuiCol_ButtonActive : hovered ? ImGuiCol_ButtonHovered : ImGuiCol_Button );
        const ImU32 col = ImGui::GetColorU32( colIdx );
        ImGui::RenderFrame( bb.Min, bb.Max, col, true, style.FrameRounding );
    }

    if ( g.LogEnabled )
        ImGui::LogSetNextTextDecoration( "[", "]" );
    StyleParamHolder sh;
    sh.addColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradBtnText ) );
    ImGui::RenderTextClipped( bb.Min, bb.Max, label, NULL, &label_size, style.ButtonTextAlign, &bb );

    IMGUI_TEST_ENGINE_ITEM_INFO( id, label, g.LastItemData.StatusFlags );
    return pressed && active;
}

bool button( const char* label, bool active, const Vector2f& size /*= Vector2f( 0, 0 )*/, ImGuiKey key /*= ImGuiKey_None */ )
{
    const ImGuiStyle& style = ImGui::GetStyle();
    const auto menu = getViewerInstance().getMenuPlugin();
    const float scaling = menu ? menu->menu_scaling() : 1.f;
    StyleParamHolder sh;
    sh.addVar( ImGuiStyleVar_FramePadding, ImVec2( style.FramePadding.x, cGradientButtonFramePadding * scaling ) );

    return buttonEx( label, active, size ) || ( active && checkKey( key ) );
}

bool buttonCommonSize( const char* label, const Vector2f& size /*= Vector2f( 0, 0 )*/, ImGuiKey key /*= ImGuiKey_None */ )
{
    return buttonEx( label, true, size ) || checkKey( key );
}

bool checkbox( const char* label, bool* value )
{
    const ImGuiStyle& style = ImGui::GetStyle();

    const auto menu = getViewerInstance().getMenuPlugin();
    const float scaling = menu ? menu->menu_scaling() : 1.f;

    StyleParamHolder sh;
    sh.addVar( ImGuiStyleVar_ItemInnerSpacing, ImVec2( cRadioInnerSpacingX * scaling, style.ItemInnerSpacing.y * scaling ) );
    auto& texture = getTexture( TextureType::Gradient );
    if ( !texture )
        return ImGui::Checkbox( label, value );

    const auto bgColor = ImGui::GetColorU32( ImGuiCol_FrameBg );

    sh.addColor( ImGuiCol_FrameBg, Color::transparent() );
    sh.addColor( ImGuiCol_CheckMark, Color::white() );
    sh.addVar( ImGuiStyleVar_FrameBorderSize, 1.5f );
    sh.addVar( ImGuiStyleVar_FramePadding, { cCheckboxPadding * scaling, cCheckboxPadding * scaling } );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
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
        if ( !ImGui::GetCurrentContext() || !v )
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
                const ImVec2 startPoint{ pos.x + ninth, pos.y + half };
                const ImVec2 anglePoint{ pos.x + half, pos.y + sz - ninth };
                const ImVec2 endPoint{ pos.x + sz - ninth, pos.y + ninth * 2.0f };

                draw_list->PathLineTo( startPoint );
                draw_list->PathLineTo( anglePoint );
                draw_list->PathLineTo( endPoint );
                draw_list->PathStroke( col, 0, thickness );

                const float radius = thickness * 0.5f;
                draw_list->AddCircleFilled( startPoint, radius, col );
                draw_list->AddCircleFilled( anglePoint, radius, col );
                draw_list->AddCircleFilled( endPoint, radius, col );
            };
            renderCustomCheckmark( window->DrawList, { check_bb.Min.x + pad, check_bb.Min.y + pad }, check_col, square_sz - pad * 2.0f );
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

    return res;
}

bool checkboxValid( const char* label, bool* value, bool valid )
{
    if ( valid )
        return checkbox( label, value );

    StyleParamHolder sh;
    const auto disColor = ImGui::GetStyleColorVec4( ImGuiCol_TextDisabled );
    sh.addColor( ImGuiCol_Text, Color( disColor.x, disColor.y, disColor.z, disColor.w ) );
    bool falseVal = false;
    checkbox( label, &falseVal );
    return false;
}

bool checkboxMixed( const char* label, bool* value, bool mixed )
{
    if ( mixed )
    {
        ImGuiContext& g = *ImGui::GetCurrentContext();
        ImGuiItemFlags backup_item_flags = g.CurrentItemFlags;
        g.CurrentItemFlags |= ImGuiItemFlags_MixedValue;
        const bool changed = UI::checkbox( label, value );
        g.CurrentItemFlags = backup_item_flags;
        return changed;
    }
    else
    {
        return UI::checkbox( label, value );
    }
}

bool radioButton( const char* label, int* value, int valButton )
{
    const ImGuiStyle& style = ImGui::GetStyle();

    const auto menu = getViewerInstance().getMenuPlugin();
    const float scaling = menu ? menu->menu_scaling() : 1.f;

    StyleParamHolder sh;
    sh.addVar( ImGuiStyleVar_ItemInnerSpacing, ImVec2( cRadioInnerSpacingX * scaling, style.ItemInnerSpacing.y * scaling ) );

    auto& texture = getTexture( TextureType::Gradient );
    if ( !texture )
    {
        const bool res = ImGui::RadioButton( label, value, valButton );
        return res;
    }

    const auto bgColor = ImGui::GetColorU32( ImGuiCol_FrameBg );

    sh.addColor( ImGuiCol_FrameBg, Color::transparent() );
    sh.addColor( ImGuiCol_CheckMark, Color::white() );
    sh.addVar( ImGuiStyleVar_FrameBorderSize, 1.0f );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;

    const float clickSize = cRadioButtonSize * scaling;

    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb( pos, ImVec2( pos.x + clickSize, pos.y + clickSize ) );

    if ( value && *value == valButton )
        ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded(
            texture->getImTextureId(),
            bb.Min, bb.Max,
            ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
            Color::white().getUInt32(), clickSize * 0.5f );

    //code of this lambda is copied from ImGui::RadioBitton in order to decrease size of the central circle
    auto drawCustomRadioButton = [bgColor, scaling, clickSize, &style] ( const char* label, int* v, int v_button )
    {
        if ( !ImGui::GetCurrentContext() || !v )
            return false;

        ImGuiContext& g = *ImGui::GetCurrentContext();
        ImGuiWindow* window = g.CurrentWindow;
        if ( !window || window->SkipItems )
            return false;

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

        const float textHeight = ImGui::GetTextLineHeight();
        const float textCenterToY = textHeight - std::ceil( textHeight / 2.f );
        ImVec2 label_pos = ImVec2( check_bb.Max.x + style.ItemInnerSpacing.x, ( check_bb.Min.y + check_bb.Max.y ) / 2.f - textCenterToY );
        ImGui::RenderText( label_pos, label );

        IMGUI_TEST_ENGINE_ITEM_INFO( id, label, g.LastItemData.StatusFlags );
        return pressed;
    };

    auto res = drawCustomRadioButton( label, value, valButton );

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

bool colorEdit4( const char* label, Vector4f& color, ImGuiColorEditFlags flags /*= ImGuiColorEditFlags_None*/ )
{
    using namespace ImGui;
    const auto& style = GetStyle();

    StyleParamHolder sh;
    sh.addVar( ImGuiStyleVar_FramePadding, { 8.f, 3.f } );
    sh.addVar( ImGuiStyleVar_FrameRounding, 2.f );
    sh.addVar( ImGuiStyleVar_ItemInnerSpacing, { cRadioInnerSpacingX, style.ItemInnerSpacing.y } );

    /// Copy of ImGui code. Required to implement own code
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
            auto& texture = getTexture( isRainbow ? TextureType::RainbowRect : TextureType::Mono );
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

    return value_changed;
}

bool colorEdit4( const char* label, Color& color, ImGuiColorEditFlags flags /*= ImGuiColorEditFlags_None */ )
{
    Vector4f color4f( color );
    const bool res = colorEdit4( label, color4f, flags );
    color = Color( color4f );
    return res;
}

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

bool combo( const char* label, int* v, const std::vector<std::string>& options, bool showPreview /*= true*/,
    const std::vector<std::string>& tooltips /*= {}*/, const std::string& defaultText /*= "Not selected" */ )
{
    assert( tooltips.empty() || tooltips.size() == options.size() );

    StyleParamHolder sh;
    const float menuScaling = Viewer::instanceRef().getMenuPlugin()->menu_scaling();
    sh.addVar( ImGuiStyleVar_FramePadding, menuScaling * StyleConsts::CustomCombo::framePadding );

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

    auto res = ImGui::BeginCombo( label, nullptr, ImGuiComboFlags_NoArrowButton );
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

    if ( !res )
        return false;

    bool selected = false;
    for ( int i = 0; i < int( options.size() ); ++i )
    {
        ImGui::PushID( ( label + std::to_string( i ) ).c_str() );
        if ( ImGui::Selectable( options[i].c_str(), *v == i ) )
        {
            selected = true;
            *v = i;
        }

        if ( !tooltips.empty() )
            UI::setTooltipIfHovered( tooltips[i], menuScaling );

        ImGui::PopID();
    }

    ImGui::EndCombo();
    if ( !showPreview )
        ImGui::PopItemWidth();
    return selected;
}

bool beginCombo( const char* label, const std::string& text /*= "Not selected" */, bool showPreview /*= true*/ )
{
    StyleParamHolder sh;
    sh.addVar( ImGuiStyleVar_FramePadding, StyleConsts::CustomCombo::framePadding );

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

    auto res = ImGui::BeginCombo( label, nullptr, ImGuiComboFlags_NoArrowButton );
    if ( showPreview )
    {
        ImGui::RenderTextClipped( { boundingBox.Min.x + style.FramePadding.x, boundingBox.Min.y + style.FramePadding.y }, { boundingBox.Max.x - arrowSize, boundingBox.Max.y }, text.c_str(), nullptr, nullptr );
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

    return res;
}

void endCombo( bool showPreview /*= true*/ )
{
    ImGui::EndCombo();
    if ( !showPreview )
        ImGui::PopItemWidth();
}

bool inputTextCentered( const char* label, std::string& str, float width /*= 0.0f*/,
    ImGuiInputTextFlags flags /*= 0*/, ImGuiInputTextCallback callback /*= NULL*/, void* user_data /*= NULL */ )
{
    const auto& style = ImGui::GetStyle();
    const auto& viewer = MR::Viewer::instanceRef();
    const auto estimatedSize = ImGui::CalcTextSize( str.c_str() );
    const float scaling = viewer.getMenuPlugin() ? viewer.getMenuPlugin()->menu_scaling() : 1.0f;
    const ImVec2 padding{ 2 * style.FramePadding.x * scaling , 2 * style.FramePadding.y * scaling };
    const auto actualWidth = ( width == 0.0f ) ? estimatedSize.x + padding.x : width;

    ImGui::SetNextItemWidth( actualWidth );
    StyleParamHolder sh;
    if ( actualWidth > estimatedSize.x )
        sh.addVar( ImGuiStyleVar_FramePadding, { ( actualWidth - estimatedSize.x ) * 0.5f, style.FramePadding.y } );

    return ImGui::InputText( label, str, flags, callback, user_data );
}

void inputTextCenteredReadOnly( const char* label, const std::string& str, float width /*= 0.0f*/, const std::optional<ImVec4>& textColor /*= {} */ )
{
    const auto& style = ImGui::GetStyle();
    const auto& viewer = MR::Viewer::instanceRef();
    const auto estimatedSize = ImGui::CalcTextSize( str.c_str() );
    const float scaling = viewer.getMenuPlugin() ? viewer.getMenuPlugin()->menu_scaling() : 1.0f;
    const ImVec2 padding{ 2 * style.FramePadding.x * scaling , 2 * style.FramePadding.y * scaling };
    const auto actualWidth = ( width == 0.0f ) ? estimatedSize.x + padding.x : width;

    ImGui::SetNextItemWidth( actualWidth );
    StyleParamHolder sh;
    if ( actualWidth > estimatedSize.x )
        sh.addVar( ImGuiStyleVar_FramePadding, { ( actualWidth - estimatedSize.x ) * 0.5f, style.FramePadding.y } );

    if ( textColor )
    {
        ImGui::PushStyleColor( ImGuiCol_Text, *textColor );
    }
    else
    {
        auto transparentColor = ImGui::GetStyleColorVec4( ImGuiCol_Text );
        transparentColor.w *= 0.5f;
        ImGui::PushStyleColor( ImGuiCol_Text, transparentColor );
    }
    ImGui::InputText( ( std::string( "##" ) + label ).c_str(), const_cast< std::string& >( str ), ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_AutoSelectAll );
    ImGui::PopStyleColor();
    ImGui::SameLine();

    if ( label && label[0] != '#' && label[0] != '\0' && label[1] != '#' )
        ImGui::Text( "%s", label );

}

void transparentText( const char* fmt, ... )
{
    auto transparentColor = ImGui::GetStyleColorVec4( ImGuiCol_Text );
    transparentColor.w *= 0.5f;
    ImGui::PushStyleColor( ImGuiCol_Text, transparentColor );
    va_list args;
    va_start( args, fmt );
    ImGui::TextV( fmt, args );
    va_end( args );
    ImGui::PopStyleColor();
}

void transparentTextWrapped( const char* fmt, ... )
{
    auto transparentColor = ImGui::GetStyleColorVec4( ImGuiCol_Text );
    transparentColor.w *= 0.5f;
    ImGui::PushStyleColor( ImGuiCol_Text, transparentColor );
    va_list args;
    va_start( args, fmt );
    ImGui::TextWrappedV( fmt, args );
    va_end( args );
    ImGui::PopStyleColor();
}

void setTooltipIfHovered( const std::string& text, float scaling )
{
    if ( !ImGui::IsItemHovered() || ImGui::IsItemActive() )
        return;
    assert( scaling > 0.f );

    // default ImGui values
    StyleParamHolder sh;
    sh.addVar( ImGuiStyleVar_FramePadding, { 4.0f * scaling, 5.0f * scaling } );
    sh.addVar( ImGuiStyleVar_WindowPadding, { 8.0f * scaling, 8.0f * scaling } );

    constexpr float cMaxWidth = 400.f;
    const auto& style = ImGui::GetStyle();
    auto textSize = ImGui::CalcTextSize( text.c_str(), nullptr, false, cMaxWidth * scaling - style.WindowPadding.x * 2 );
    ImGui::SetNextWindowSize( ImVec2{ textSize.x + style.WindowPadding.x * 2, 0 } );

    ImGui::BeginTooltip();
    ImGui::TextWrapped( "%s", text.c_str() );
    ImGui::EndTooltip();
}

void separator( float scaling, const std::string& text /*= ""*/, int issueCount /*= -1 */ )
{
    const auto& style = ImGui::GetStyle();
    if ( style.ItemSpacing.y < MR::cSeparateBlocksSpacing * scaling )
    {
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + MR::cSeparateBlocksSpacing * scaling );
    }
    
    if ( text.empty() )
    {
        ImGui::Separator();
    }
    else if ( ImGui::BeginTable( (std::string("SeparatorTable_") + text).c_str(), 2, ImGuiTableFlags_SizingFixedFit ) )
    {
        ImGui::TableNextColumn();
        ImGui::PushFont( MR::RibbonFontManager::getFontByTypeStatic( MR::RibbonFontManager::FontType::SemiBold ) );
        ImGui::Text( "%s", text.c_str());
        ImGui::SameLine();
        if ( issueCount >= 0 )
        {
            ImGui::PushStyleColor( ImGuiCol_FrameBg, issueCount > 0 ? ImVec4{ 0.886f, 0.267f, 0.267f, 1.0f} : ImVec4{ 0.235f, 0.663f, 0.078f, 1.0f } );            
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() - ImGui::GetTextLineHeight() * 0.5f + style.FramePadding.y * 0.5f );
            const std::string issue = std::to_string( issueCount );
            const float width = std::max( 20.0f * scaling, ImGui::CalcTextSize( issue.data() ).x + 2.0f * style.FramePadding.x );
            UI::inputTextCenteredReadOnly( "##IssueCount", issue, width, ImGui::GetStyleColorVec4(ImGuiCol_Text) );
            ImGui::PopStyleColor();
        }
        ImGui::PopFont();

        ImGui::TableNextColumn();
        auto width = ImGui::GetWindowWidth();
        ImGui::SetCursorPos( { width - ImGui::GetStyle().WindowPadding.x, ImGui::GetCursorPosY() + std::round(ImGui::GetTextLineHeight() * 0.5f) } );
        ImGui::Separator();
        ImGui::EndTable();
    }

    if ( ImGui::GetStyle().ItemSpacing.y < MR::cSeparateBlocksSpacing * scaling )
    {
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + MR::cSeparateBlocksSpacing * scaling - ImGui::GetStyle().ItemSpacing.y );
    }
    ImGui::Dummy( ImVec2( 0, 0 ) );
}

void progressBar( float scaling, float fraction, const Vector2f& sizeArg /*= Vector2f( -1, 0 ) */ )
{
    auto& textureG = getTexture( TextureType::Gradient );
    if ( !textureG )
        return ImGui::ProgressBar( fraction, sizeArg );
    auto* context = ImGui::GetCurrentContext();
    if ( !context )
        return;
    ImGuiWindow* window = context->CurrentWindow;
    if ( !window || window->SkipItems )
        return;

    auto* drawList = window->DrawList;
    if ( !drawList )
        return;

    const ImGuiStyle& style = context->Style;

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 size = ImGui::CalcItemSize( sizeArg, ImGui::CalcItemWidth(), ImGui::GetFrameHeight() );
    ImRect bb( pos, pos + size );
    ImGui::ItemSize( size, style.FramePadding.y );
    if ( !ImGui::ItemAdd( bb, 0 ) )
        return;

    auto textWidth = ImGui::CalcTextSize( "65%" ).x; // text given for reference in design

    auto pgWidth = size.x - textWidth - StyleConsts::ProgressBar::internalSpacing * scaling;

    const auto& bgColor = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ProgressBarBackground );
    drawList->AddRectFilled( bb.Min, ImVec2( bb.Min.x + pgWidth, bb.Max.y ), bgColor.getUInt32(), StyleConsts::ProgressBar::rounding * scaling );
    if ( fraction > 0.0f )
    {
        drawList->AddImageRounded(
            textureG->getImTextureId(),
            bb.Min,
            ImVec2( bb.Min.x + pgWidth * std::clamp( fraction, 0.0f, 1.0f ), bb.Max.y ),
            ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
            Color::white().getUInt32(), StyleConsts::ProgressBar::rounding * scaling );
    }
    // Default displaying the fraction as percentage string, but user can override it
    char textBuf[8];
    ImFormatString( textBuf, IM_ARRAYSIZE( textBuf ), "%d%%", int( fraction * 100 ) );
    ImVec2 realTextSize = ImGui::CalcTextSize( textBuf );

    ImGui::RenderText( ImVec2( bb.Max.x - realTextSize.x, bb.Min.y + ( size.y - realTextSize.y ) * 0.5f ), textBuf );
}

} // namespace UI

}
