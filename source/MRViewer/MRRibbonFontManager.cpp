#include "MRRibbonFontManager.h"
#include "misc/freetype/imgui_freetype.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRRibbonConstants.h"
#include "imgui_fonts_droid_sans.h"
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include "MRRibbonMenu.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

static void loadFontChecked( const char* filename, float size_pixels, const ImFontConfig* font_cfg = nullptr, const ImWchar* glyph_ranges = nullptr )
{
    if ( !ImGui::GetIO().Fonts->AddFontFromFileTTF( filename, size_pixels, font_cfg, glyph_ranges ) )
    {
        assert( false && "Failed to load font!" );
        spdlog::error( "Failed to load font from `{}`.", filename );

        ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
            droid_sans_compressed_size, size_pixels, font_cfg, glyph_ranges );
    }
}

void RibbonFontManager::loadAllFonts( ImWchar* charRanges, float scaling )
{
    fonts_ = {};

    const ImWchar iconRanges[] = { 0xe005, 0xf8ff, 0 };

    std::vector<CustomGlyph> customGlyphs;

    for ( int i = 0; i< int( FontType::Count ); ++i )
    {
        if ( i == int( FontType::Monospace ) )
            loadFont_( FontType::Monospace, ImGui::GetIO().Fonts->GetGlyphRangesDefault(), scaling );
        if ( i == int( FontType::Icons ) )
            loadFont_( FontType::Icons, iconRanges, scaling );
        else
            loadFont_( FontType( i ), charRanges, scaling );

        addCustomGlyphs_( FontType( i ), scaling, customGlyphs );
    }
    ImGui::GetIO().Fonts->Build();

    renderCustomGlyphsToAtlas_( customGlyphs );
}

ImFont* RibbonFontManager::getFontByType( FontType type ) const
{
    return fonts_[int( type )];
}

float RibbonFontManager::getFontSizeByType( FontType type )
{
    switch ( type )
    {
    case MR::RibbonFontManager::FontType::Default:
    case MR::RibbonFontManager::FontType::SemiBold:
    case MR::RibbonFontManager::FontType::Monospace:
        return cDefaultFontSize;
    case MR::RibbonFontManager::FontType::Small:
        return cSmallFontSize;
    case MR::RibbonFontManager::FontType::Icons:
        return cBigIconSize;
    case MR::RibbonFontManager::FontType::Headline:
        return cHeadlineFontSize;
    case MR::RibbonFontManager::FontType::Big:
    case MR::RibbonFontManager::FontType::BigSemiBold:
        return cBigFontSize;
    case MR::RibbonFontManager::FontType::Count:
        break;
    }

    assert( false && "Unknown font enum!" );
    return 0;
}

std::filesystem::path RibbonFontManager::getMenuFontPath() const
{
#ifndef __EMSCRIPTEN__
    return  GetFontsDirectory() / "NotoSansSC-Regular.otf";
#else
    return  GetFontsDirectory() / "NotoSans-Regular.ttf";
#endif
}

ImFont* RibbonFontManager::getFontByTypeStatic( FontType type )
{
    RibbonFontManager* fontManager = getFontManagerInstance_();
    if ( fontManager )
        return fontManager->getFontByType( type );
    return nullptr;
}

void RibbonFontManager::initFontManagerInstance( RibbonFontManager* ribbonFontManager )
{
    getFontManagerInstance_() = ribbonFontManager;
}

std::filesystem::path RibbonFontManager::getMenuLatinSemiBoldFontPath_() const
{
    return getMenuFontPath().parent_path() / "NotoSans-SemiBold.ttf";
}

MR::RibbonFontManager*& RibbonFontManager::getFontManagerInstance_()
{
    static RibbonFontManager* instance{ nullptr };
    return instance;
}

void RibbonFontManager::loadFont_( FontType type, const ImWchar* ranges, float scaling )
{
    float fontSize = getFontSizeByType( type ) * scaling;

    if ( type == FontType::Default )
    {
        auto fontPath = getMenuFontPath();
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
#ifndef __EMSCRIPTEN__
        config.GlyphOffset = ImVec2( 0, -4 * scaling );
#else
        config.GlyphOffset = ImVec2( 0, -3 * scaling );
#endif
        loadFontChecked(
            utf8string( fontPath ).c_str(), fontSize,
            &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::Icons )
    {
        ImFontConfig config;
        config.GlyphMinAdvanceX = fontSize; // Use if you want to make the icon monospaced
        auto fontPath = GetFontsDirectory() / "fa-solid-900.ttf";
        loadFontChecked( utf8string( fontPath ).c_str(), fontSize, &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::Small )
    {
        auto fontPath = getMenuFontPath();
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
#ifndef __EMSCRIPTEN__
        config.GlyphOffset = ImVec2( 0, -3 * scaling );
#else
        config.GlyphOffset = ImVec2( 0, -2 * scaling );
#endif
        loadFontChecked(
            utf8string( fontPath ).c_str(), fontSize,
            &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::SemiBold )
    {
        auto fontPath = getMenuLatinSemiBoldFontPath_();
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
        // "- 3 * scaling" eliminates shift of the font in order to render this font in text fields properly
        config.GlyphOffset = ImVec2( 0, - 3 * scaling );
        loadFontChecked(
            utf8string( fontPath ).c_str(), fontSize,
            &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::Big )
    {
        auto fontPath = getMenuFontPath();
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
        config.GlyphOffset = ImVec2( 0, -4 * scaling );
        loadFontChecked(
            utf8string( fontPath ).c_str(), fontSize,
            &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::BigSemiBold )
    {
        auto fontPath = getMenuLatinSemiBoldFontPath_();
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
        config.GlyphOffset = ImVec2( 0, -4 * scaling );
        loadFontChecked(
            utf8string( fontPath ).c_str(), fontSize,
            &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::Headline )
    {
        auto fontPath = getMenuLatinSemiBoldFontPath_();
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
        config.GlyphOffset = ImVec2( 0, -4 * scaling );
        loadFontChecked(
            utf8string( fontPath ).c_str(), fontSize,
            &config, ranges );
        fonts_[int( type )] = ImGui::GetIO().Fonts->Fonts.back();
    }
    else if ( type == FontType::Monospace )
    {
        auto fontPath = GetFontsDirectory() / "NotoSansMono-Regular.ttf";
        ImFontConfig config;
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
        config.GlyphOffset = ImVec2( 1 * scaling, -2 * scaling );
        loadFontChecked(
            utf8string( fontPath ).c_str(), fontSize,
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

void RibbonFontManager::addCustomGlyphs_( FontType font, float scaling, std::vector<CustomGlyph>& glyphs )
{
    // `font->FontSize` is null at this point, so we must pass `fontSize` manually.

    auto addGlyph = [&](
        ImWchar ch, float relWidth,
        std::function<void( unsigned char* texture, int stride, int rectW, int rectH )> render
    )
    {
        int height = int( std::floor( getFontSizeByType( font ) * scaling ) );
        int width = int( std::round( height * relWidth ) );

        int index = ImGui::GetIO().Fonts->AddCustomRectFontGlyph( fonts_[int( font )], ch, width, height, float( width ) );
        auto renderWrapper = [index, func = std::move( render )]( unsigned char* texData, int texW )
        {
            const ImFontAtlasCustomRect* rect = ImGui::GetIO().Fonts->GetCustomRectByIndex(index);
            func( texData + rect->X + rect->Y * texW, texW, rect->Width, rect->Height );
        };
        glyphs.push_back( CustomGlyph{ .render = renderWrapper } );
    };

    if ( font != FontType::Icons )
    {
        addGlyph( 0x207B /*SUPERSCRIPT MINUS*/, 0.25f, []( unsigned char* texture, int stride, int rectW, int rectH )
        {
            int lineH = int( rectH * 0.30f );

            for ( int y = 0; y < rectH; y++ )
            {
                unsigned char value = y == lineH ? 255 : 0;
                for ( int x = 0; x < rectW; x++ )
                    texture[x + y * stride] = value;
            }
        } );
    }
}

void RibbonFontManager::renderCustomGlyphsToAtlas_( const std::vector<CustomGlyph>& glyphs )
{
    unsigned char* texData = nullptr;
    int texW = 0;
    ImGui::GetIO().Fonts->GetTexDataAsAlpha8( &texData, &texW, nullptr );
    for ( const CustomGlyph& glyph : glyphs )
        glyph.render( texData, texW );
}

}
