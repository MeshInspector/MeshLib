#include "MRRibbonFontManager.h"
#include "misc/freetype/imgui_freetype.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"
#include "MRRibbonConstants.h"
#include "imgui_fonts_droid_sans.h"
#include "MRRibbonMenu.h"
#include "MRPch/MRSpdlog.h"
#include "MRCommandLoop.h"
#include "backends/imgui_impl_opengl3.h"

namespace MR
{

static ImFont* loadFontChecked( const char* filename, float size_pixels, const ImFontConfig* font_cfg = nullptr, const ImWchar* glyph_ranges = nullptr )
{
    auto font = ImGui::GetIO().Fonts->AddFontFromFileTTF( filename, size_pixels, font_cfg, glyph_ranges );
    if ( !font )
    {
        assert( false && "Failed to load font!" );
        spdlog::error( "Failed to load font from `{}`.", filename );

        font =ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
            droid_sans_compressed_size, size_pixels, font_cfg, glyph_ranges );
    }
    return font;
}

RibbonFontManager::RibbonFontManager()
{
    fontPaths_ =
    {
#ifndef __EMSCRIPTEN__
    SystemPath::getFontsDirectory() / "NotoSansSC-Regular.otf",
#else
    SystemPath::getFontsDirectory() / "NotoSans-Regular.ttf",
#endif
    SystemPath::getFontsDirectory() / "NotoSans-SemiBold.ttf",
    SystemPath::getFontsDirectory() / "NotoSansMono-Regular.ttf",
    SystemPath::getFontsDirectory() / "fa-solid-900.ttf"
    };
}

void RibbonFontManager::loadAllFonts( ImWchar* charRanges, float scaling )
{
    fonts_ = {
        FontData{.fontFile = FontFile::Regular},
        FontData{.fontFile = FontFile::Regular},
        FontData{.fontFile = FontFile::SemiBold},
        FontData{.fontFile = FontFile::Icons},
        FontData{.fontFile = FontFile::Regular},
        FontData{.fontFile = FontFile::SemiBold},
        FontData{.fontFile = FontFile::SemiBold},
        FontData{.fontFile = FontFile::Monospace}
    };

    updateFontsScaledOffset_( scaling );

    const ImWchar iconRanges[] = { 0xe005, 0xf8ff, 0 };

    std::vector<CustomGlyph> customGlyphs;

    for ( int i = 0; i< int( FontType::Count ); ++i )
    {
        if ( i == int( FontType::Monospace ) )
            loadFont_( FontType::Monospace, ImGui::GetIO().Fonts->GetGlyphRangesDefault(), scaling );
        else if ( i == int( FontType::Icons ) )
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
    return fonts_[int( type )].fontPtr;
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
    return fontPaths_[int( FontFile::Regular )];
}

void RibbonFontManager::setNewFontPaths( const FontFilePaths& paths )
{
    fontPaths_ = paths;
    if ( auto menu = ImGuiMenu::instance() )
    {
        CommandLoop::appendCommand( [menu] ()
        {
            menu->reload_font();
            ImGui_ImplOpenGL3_DestroyDeviceObjects(); // needed to update font
        } );
    }
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

MR::RibbonFontManager*& RibbonFontManager::getFontManagerInstance_()
{
    static RibbonFontManager* instance{ nullptr };
    return instance;
}

void RibbonFontManager::updateFontsScaledOffset_( float scaling )
{
    ImGuiIO& io = ImGui::GetIO();
    const ImWchar wRange[] = { 0x0057, 0x0057, 0 }; // `W` symbol
    std::array<ImFont*, int( FontType::Count )> localFonts;
    for ( int i = 0; i < int( FontType::Count ); ++i )
    {
        auto& font = fonts_[int( i )];
        auto fontPath = fontPaths_[int( font.fontFile )];

        ImFontConfig config;
        if ( i != int( FontType::Icons ) )
            config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;

        auto fontSize = getFontSizeByType( FontType( i ) ) * scaling;
        localFonts[i] = io.Fonts->AddFontFromFileTTF( utf8string( fontPath ).c_str(), fontSize, &config, wRange );
    }
    io.Fonts->Build();
    for ( int i = 0; i < int( FontType::Count ); ++i )
    {
        auto* lFont = localFonts[i];
        if ( !lFont )
            continue;
        if ( lFont->Glyphs.size() != 1 )
            continue;
        const auto& glyph = lFont->Glyphs.back();

        auto& fontRef = fonts_[int( i )];
        auto fontSize = getFontSizeByType( FontType( i ) ) * scaling;
        Box2f box;
        box.include( Vector2f( glyph.X0, glyph.Y0 ) );
        box.include( Vector2f( glyph.X1, glyph.Y1 ) );
        fontRef.scaledOffset = 0.5f * ( Vector2f::diagonal( fontSize ) - box.size() ) - box.min;
        fontRef.scaledOffset.x = std::round( -box.min.x ); // looks like Dear ImGui expecting glyph to start at the left side of the box, and not being in the center
        fontRef.scaledOffset.y = std::round( fontRef.scaledOffset.y );
    }
    io.Fonts->Clear();
}

void RibbonFontManager::loadFont_( FontType type, const ImWchar* ranges, float scaling )
{
    float fontSize = getFontSizeByType( type ) * scaling;
    auto& font = fonts_[int( type )];
    auto fontPath = fontPaths_[int( font.fontFile )];

    ImFontConfig config;
    if ( type == FontType::Icons )
    {
        config.GlyphMinAdvanceX = fontSize; // Use if you want to make the icon monospaced
    }
    else
    {
        config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_Bitmap;
        config.GlyphOffset = ImVec2( font.scaledOffset );
    }

    font.fontPtr = loadFontChecked(
        utf8string( fontPath ).c_str(), fontSize,
        &config, ranges );
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

        int index = ImGui::GetIO().Fonts->AddCustomRectFontGlyph( fonts_[int( font )].fontPtr, ch, width, height, float( width ) );
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
