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

    MRVIEWER_API RibbonFontManager();

    /// load all fonts using in ribbon menu
    MRVIEWER_API void loadAllFonts( ImWchar* charRanges, float scaling );

    /// get font by font type
    MRVIEWER_API ImFont* getFontByType( FontType type ) const;
    /// get font size by font type
    MRVIEWER_API static float getFontSizeByType( FontType type );

    /// get ribbon menu font path
    MRVIEWER_API std::filesystem::path getMenuFontPath() const;

    /// get font by font type
    /// (need to avoid dynamic cast menu to ribbon menu)
    MRVIEWER_API static ImFont* getFontByTypeStatic( FontType type );

    /// initialize static holder for easier access to ribbon fonts
    /// (need to avoid dynamic cast menu to ribbon menu)
    MRVIEWER_API static void initFontManagerInstance( RibbonFontManager* ribbonFontManager );

private:
    std::array<ImFont*, size_t( FontType::Count )> fonts_{ nullptr,nullptr,nullptr,nullptr };
    std::array<std::filesystem::path, size_t( UniqueFont::Count )> fontPaths_;
    std::array<UniqueFont, size_t( FontType::Count )> fontTypeMap_{ 
        UniqueFont::Regular, 
        UniqueFont::Regular, 
        UniqueFont::SemiBold, 
        UniqueFont::Icons,
        UniqueFont::Regular, 
        UniqueFont::SemiBold,
        UniqueFont::SemiBold,
        UniqueFont::Monospace };

    /// get pointer to instance of this class (if it exists)
    static RibbonFontManager*& getFontManagerInstance_();

    void loadFont_( FontType type, const ImWchar* ranges, float scaling );

    struct CustomGlyph
    {
        std::function<void( unsigned char* texData, int texW )> render;
    };
    void addCustomGlyphs_( FontType font, float scaling, std::vector<CustomGlyph>& glyphs );
    void renderCustomGlyphsToAtlas_( const std::vector<CustomGlyph>& glyphs );
};

}
