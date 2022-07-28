#include "MRRibbonFontManager.h"
#include "misc/freetype/imgui_freetype.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRRibbonConstants.h"
#include "imgui_fonts_droid_sans.h"

namespace MR
{

void RibbonFontManager::loadAllFonts( ImWchar* charRanges, float scaling )
{
    fonts_ = { nullptr,nullptr,nullptr,nullptr };

    const ImWchar iconRanges[] = { 0xe005, 0xf8ff, 0 };

    for ( int i = 0; i<int( FontType::Count ); ++i )
    {
        if ( i != int( FontType::Icons ) )
            loadFont_( FontType( i ), charRanges, scaling );
        else
            loadFont_( FontType::Icons, iconRanges, scaling );
    }
    ImGui::GetIO().Fonts->Build();
}

ImFont* RibbonFontManager::getFontByType( FontType type ) const
{
    return fonts_[int( type )];
}

float RibbonFontManager::getFontSizeByType( FontType type ) const
{
    switch ( type )
    {
    case MR::RibbonFontManager::FontType::Default:
    case MR::RibbonFontManager::FontType::SemiBold:
        return cDefaultFontSize;
    case MR::RibbonFontManager::FontType::Small:
        return cSmallFontSize;
    case MR::RibbonFontManager::FontType::Icons:
        return cBigIconSize;
    case MR::RibbonFontManager::FontType::Count:
    default:
        return 0.f;
        break;
    }
}

std::filesystem::path RibbonFontManager::getMenuFontPath() const
{
    return  GetFontsDirectory() / "selawk.ttf";
}

void RibbonFontManager::loadFont_( FontType type, const ImWchar* ranges, float scaling )
{
    if ( type == FontType::Default )
    {
        auto fontPath = getMenuFontPath();
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
        config.GlyphOffset = ImVec2( 0, -2 * scaling );
        ImGui::GetIO().Fonts->AddFontFromFileTTF(
            utf8string( fontPath ).c_str(), cDefaultFontSize * scaling,
            &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::Icons )
    {
        ImFontConfig config;
        const float fontSize = cBigIconSize * scaling;
        config.GlyphMinAdvanceX = fontSize; // Use if you want to make the icon monospaced
        auto fontPath = GetFontsDirectory() / "fa-solid-900.ttf";
        ImGui::GetIO().Fonts->AddFontFromFileTTF( utf8string( fontPath ).c_str(), fontSize, &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::Small )
    {
        auto fontPath = getMenuFontPath();
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
        config.GlyphOffset = ImVec2( 0, -2 * scaling );
        ImGui::GetIO().Fonts->AddFontFromFileTTF(
            utf8string( fontPath ).c_str(), cSmallFontSize * scaling,
            &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::SemiBold )
    {
        auto fontPath = getMenuFontPath();
        fontPath = fontPath.parent_path() / "selawksb.ttf";
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
        config.GlyphOffset = ImVec2( 0, 1 * scaling );
        ImGui::GetIO().Fonts->AddFontFromFileTTF(
            utf8string( fontPath ).c_str(), cDefaultFontSize * scaling,
            &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::Big )
    {
        auto fontPath = getMenuFontPath();
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
        config.GlyphOffset = ImVec2( 0, -3 * scaling );
        ImGui::GetIO().Fonts->AddFontFromFileTTF(
            utf8string( fontPath ).c_str(), cBigFontSize * scaling,
            &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
}

void RibbonFontManager::loadDefaultFont_( float fontSize, float yOffset )
{
    ImFontConfig config;
    config.GlyphOffset = ImVec2( 0, yOffset );
#ifndef __EMSCRIPTEN__
    config.OversampleH = 7;
    config.OversampleV = 7;
#endif
    ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
                                                          droid_sans_compressed_size, fontSize,
                                                          &config);
}

}
