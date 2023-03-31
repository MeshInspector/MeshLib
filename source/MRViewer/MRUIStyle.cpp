#include "MRUIStyle.h"
#include "MRImGuiImage.h"
#include "MRRibbonButtonDrawer.h"
#include "MRColorTheme.h"
#include "MRRibbonConstants.h"
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include "imgui_internal.h"
#include "ImGuiHelpers.h"

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

    auto res = ImGui::ButtonValid( label, active, size );

    ImGui::PopStyleVar( pushedStyleNum );
    ImGui::PopStyleColor( 2 );
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

    int pushedStyleNum = 1;
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );

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

    return res;
}

} // namespace UI

}
