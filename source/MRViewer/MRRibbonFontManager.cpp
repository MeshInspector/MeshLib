#include "MRRibbonFontManager.h"
#include "MRViewer/MRUIStyle.h"
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

static ImFont* loadFontChecked( const char* filename, float size_pixels, const ImFontConfig* font_cfg = nullptr, const ImWchar* glyph_ranges = nullptr, const char* additionalFilename = nullptr )
{
    auto font = ImGui::GetIO().Fonts->AddFontFromFileTTF( filename, size_pixels, font_cfg, glyph_ranges );
    if ( !font )
    {
        assert( false && "Failed to load font!" );
        spdlog::error( "Failed to load font from `{}`.", filename );

        font = ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
            droid_sans_compressed_size, size_pixels, font_cfg, glyph_ranges );
    }
    if ( additionalFilename )
    {
        ImFontConfig cfg = *font_cfg;
        cfg.MergeMode = true;
        ImGui::GetIO().Fonts->AddFontFromFileTTF( additionalFilename, size_pixels, &cfg, glyph_ranges );
    }
    return font;
}

RibbonFontManager::RibbonFontManager()
{
    fontPaths_ =
    {
    SystemPath::getFontsDirectory() / "NotoSans-Regular.ttf",
#ifndef __EMSCRIPTEN__
    SystemPath::getFontsDirectory() / "NotoSansSC-Regular.otf",
#endif
    SystemPath::getFontsDirectory() / "NotoSans-SemiBold.ttf",
    SystemPath::getFontsDirectory() / "NotoSansMono-Regular.ttf",
    SystemPath::getFontsDirectory() / "fa-solid-900.ttf",
    };
}

void RibbonFontManager::loadAllFonts( ImWchar* charRanges )
{
    fonts_ = {
        FontData{.fontFile = cFontFileRegular_},
        FontData{.fontFile = cFontFileRegular_},
        FontData{.fontFile = FontFile::SemiBold},
        FontData{.fontFile = FontFile::Icons},
        FontData{.fontFile = cFontFileRegular_},
        FontData{.fontFile = FontFile::SemiBold},
        FontData{.fontFile = FontFile::SemiBold},
        FontData{.fontFile = FontFile::Monospace}
    };

    updateFontsScaledOffset_();

    const ImWchar iconRanges[] = { 0xe005, 0xf8ff, 0 };

    for ( int i = 0; i< int( FontType::Count ); ++i )
    {
        if ( i == int( FontType::Icons ) )
            loadFont_( FontType::Icons, iconRanges );
        else
            loadFont_( FontType( i ), charRanges );
    }
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
    return fontPaths_[int( cFontFileRegular_ )];
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

FontAndSize RibbonFontManager::getFontAndSizeByTypeStatic( FontType type )
{
    RibbonFontManager* fontManager = getFontManagerInstance_();
    if ( !fontManager )
        return { nullptr, 0.f };
    return { fontManager->getFontByType( type ), fontManager->getFontSizeByType( type ) };
}

void RibbonFontManager::initFontManagerInstance( RibbonFontManager* ribbonFontManager )
{
    getFontManagerInstance_() = ribbonFontManager;
}

RibbonFontManager*& RibbonFontManager::getFontManagerInstance_()
{
    static RibbonFontManager* instance{ nullptr };
    return instance;
}

void RibbonFontManager::updateFontsScaledOffset_()
{
    ImGuiIO& io = ImGui::GetIO();
    //const ImWchar wRange[] = { 0x0057, 0x0057, 0 }; // `W` symbol
    std::array<ImFont*, int( FontType::Count )> localFonts{};
    for ( int i = 0; i < int( FontType::Count ); ++i )
    {
        auto& font = fonts_[int( i )];
        auto fontPath = fontPaths_[int( font.fontFile )];

        ImFontConfig config;
        config.FontLoaderFlags = ImGuiFreeTypeLoaderFlags_Bitmap;
        if ( i == int( FontType::Icons ) )
            continue; // skip icons, because AddFontFromFileTTF return a font without glyphs, after that, io.Fonts->Build() trigger assert and crash (after update ImGui to 1.91.9)

        auto fontSize = getFontSizeByType( FontType( i ) );
        localFonts[i] = io.Fonts->AddFontFromFileTTF( utf8string( fontPath ).c_str(), fontSize, &config );

        auto* lFont = localFonts[i];
        const char wChar[] = "W\0";
        [[maybe_unused]] auto textSize = lFont->CalcTextSizeA( fontSize, 10, 10, wChar, wChar + 1 );
        auto glyph = lFont->GetFontBaked( fontSize )->FindGlyph( 'W' );
        Box2f box;
        box.include( Vector2f( glyph->X0, glyph->Y0 ) );
        box.include( Vector2f( glyph->X1, glyph->Y1 ) );
        font.scaledOffset = 0.5f * ( Vector2f::diagonal( fontSize ) - box.size() ) - box.min;
        font.scaledOffset.x = std::round( -box.min.x ); // looks like Dear ImGui expecting glyph to start at the left side of the box, and not being in the center
        font.scaledOffset.y = std::round( font.scaledOffset.y );
    }
    io.Fonts->Clear();
}

void RibbonFontManager::loadFont_( FontType type, const ImWchar* )
{
    float fontSize = getFontSizeByType( type );
    auto& font = fonts_[int( type )];
    auto fontPath = fontPaths_[int( font.fontFile )];

    ImFontConfig config;
    if ( type == FontType::Icons )
    {
        config.GlyphMinAdvanceX = fontSize; // Use if you want to make the icon monospaced
    }
    else
    {
        config.FontLoaderFlags = ImGuiFreeTypeLoaderFlags_Bitmap;
        config.GlyphOffset = ImVec2( font.scaledOffset );
    }

    bool addFont = false;
#ifndef __EMSCRIPTEN__
    addFont = font.fontFile == FontFile::RegularSC;
#endif // !__EMSCRIPTEN__
    font.fontPtr = loadFontChecked(
        utf8string( fontPath ).c_str(), fontSize,
        &config, nullptr, addFont ? utf8string( fontPaths_[0] ).c_str() : nullptr );
}

}
