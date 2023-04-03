#include "MRUIStyle.h"
#include "MRImGuiImage.h"
#include "MRRibbonButtonDrawer.h"
#include "MRColorTheme.h"
#include "MRRibbonConstants.h"
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include "ImGuiHelpers.h"
#include "ImGuiMenu.h"
#include "imgui_internal.h"

namespace MR
{


namespace UI
{

enum class TextureType
{
    Mono,
    Gradient,
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

bool button( const char* label, bool active, const Vector2f& size /*= Vector2f( 0, 0 )*/)
{
    auto& texture = getTexture( TextureType::Gradient );
    if ( !texture )
        return ImGui::ButtonValid( label, active, size );

    StyleParamHolder sh;
    sh.addColor( ImGuiCol_Button, Color::transparent() );
    sh.addColor( ImGuiCol_Text, Color::white() );

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 labelSize = ImGui::CalcTextSize( label, NULL, true );

    sh.addVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    if ( size.y == 0 )
    {
        auto framePadding = style.FramePadding;
        framePadding.y = cGradientButtonFramePadding;
        if ( auto menu = getViewerInstance().getMenuPlugin() )
            framePadding.y *= menu->menu_scaling();
        sh.addVar( ImGuiStyleVar_FramePadding, framePadding );
    }

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 realSize = ImGui::CalcItemSize( size, labelSize.x + style.FramePadding.x * 2.0f, labelSize.y + style.FramePadding.y * 2.0f );
    const ImRect bb( pos, ImVec2( pos.x + realSize.x, pos.y + realSize.y ) );

    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded(
        texture->getImTextureId(),
        bb.Min, bb.Max,
        ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
        Color::white().getUInt32(), style.FrameRounding );

    auto res = ImGui::ButtonValid( label, active, size );
        
    return res;
}

bool button( const char* label, const Vector2f& size /*= Vector2f( 0, 0 )*/, ImGuiKey key /*= ImGuiKey_None */ )
{
    auto& texture = getTexture( TextureType::Gradient );
    auto checkKey = [] ( ImGuiKey passedKey )
    {
        if ( passedKey == ImGuiKey_None )
            return false;
        if ( passedKey == ImGuiKey_Enter || passedKey == ImGuiKey_KeypadEnter )
            return ImGui::IsKeyPressed( ImGuiKey_Enter ) || ImGui::IsKeyPressed( ImGuiKey_KeypadEnter );
        return ImGui::IsKeyPressed( passedKey );
    };

    
    if ( !texture )
        return ImGui::Button( label, ImVec2( size ) ) || checkKey( key );

    StyleParamHolder sh;
    sh.addColor( ImGuiCol_Button, Color::transparent() );
    sh.addColor( ImGuiCol_Text, Color::white() );

    const auto menu = getViewerInstance().getMenuPlugin();
    const float scaling = menu ? menu->menu_scaling() : 1.f;


    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 labelSize = ImGui::CalcTextSize( label, NULL, true );

    sh.addVar( ImGuiStyleVar_FrameBorderSize, 0.0f );

    auto framePadding = style.FramePadding;
    if ( size.y == 0 )
        framePadding.y = cGradientButtonFramePadding * scaling;
    else if ( size.y > 0 )
    {
        framePadding.y = ( size.y - ImGui::CalcTextSize( label ).y ) / 2.f;
    }
    if ( size.x > 0 )
    {
        framePadding.x = ( size.x - ImGui::CalcTextSize( label ).x ) / 2.f;
    }
    sh.addVar( ImGuiStyleVar_FramePadding, framePadding );

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 realSize = ImGui::CalcItemSize( size, labelSize.x + style.FramePadding.x * 2.0f, labelSize.y + style.FramePadding.y * 2.0f );
    const ImRect bb( pos, ImVec2( pos.x + realSize.x, pos.y + realSize.y ) );

    ImGui::GetCurrentContext()->CurrentWindow->DrawList->AddImageRounded(
        texture->getImTextureId(),
        bb.Min, bb.Max,
        ImVec2( 0.5f, 0.25f ), ImVec2( 0.5f, 0.75f ),
        Color::white().getUInt32(), style.FrameRounding );

    auto res = ImGui::Button( label, size ) || checkKey( key );

    return res;
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

} // namespace UI

}
