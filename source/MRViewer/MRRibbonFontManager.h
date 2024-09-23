#pragma once
#include "imgui.h"
#include <array>
#include "exports.h"
#include <filesystem>

namespace MR
{

class MRVIEWER_CLASS RibbonFontManager
{
public:

    // Font types used in current design
    enum class FontType
    {
        Default,
        Small,
        SemiBold,
        Icons,
        Big,
        BigSemiBold,
        Headline,
        Monospace,
        Count
    };

    // Unique fonts that are used for different FontTypes
    enum class UniqueFont
    {
        Regular,
        SemiBold,
        Monospace,
        Icons,
        Count
    };

    using UniqueFontPaths = std::array<std::filesystem::path, size_t( UniqueFont::Count )>;

    MRVIEWER_API RibbonFontManager();

    /// load all fonts using in ribbon menu
    MRVIEWER_API void loadAllFonts( ImWchar* charRanges, float scaling );

    /// get font by font type
    MRVIEWER_API ImFont* getFontByType( FontType type ) const;
    /// get font size by font type
    MRVIEWER_API static float getFontSizeByType( FontType type );

    /// get ribbon menu font path
    MRVIEWER_API std::filesystem::path getMenuFontPath() const;

    /// returns list of all font paths
    const UniqueFontPaths& getAllFontPaths() const { return fontPaths_; }

    /// sets new fonts paths
    /// note that it will trigger reload font
    MRVIEWER_API void setNewFontPaths( const UniqueFontPaths& paths );

    /// get font by font type
    /// (need to avoid dynamic cast menu to ribbon menu)
    MRVIEWER_API static ImFont* getFontByTypeStatic( FontType type );

    /// initialize static holder for easier access to ribbon fonts
    /// (need to avoid dynamic cast menu to ribbon menu)
    MRVIEWER_API static void initFontManagerInstance( RibbonFontManager* ribbonFontManager );

private:
    UniqueFontPaths fontPaths_;
    struct FontData
    {
        UniqueFont uniqueFontType{ UniqueFont::Regular };
        Vector2f scaledOffset;
        ImFont* fontPtr{ nullptr };
    };
    std::array<FontData, size_t( FontType::Count )> fonts_;

    /// get pointer to instance of this class (if it exists)
    static RibbonFontManager*& getFontManagerInstance_();

    /// calculates font glyph shift
    void updateFontsScaledOffset_( float scaling );

    void loadFont_( FontType type, const ImWchar* ranges, float scaling );

    struct CustomGlyph
    {
        std::function<void( unsigned char* texData, int texW )> render;
    };
    void addCustomGlyphs_( FontType font, float scaling, std::vector<CustomGlyph>& glyphs );
    void renderCustomGlyphsToAtlas_( const std::vector<CustomGlyph>& glyphs );
};

}
