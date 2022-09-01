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
        Count
    };

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

    /// get pointer to instance of this class (if it exists)
    static RibbonFontManager*& getFontManagerInstance_();

    void loadFont_( FontType type, const ImWchar* ranges, float scaling );

    /// load default font (droid_sans)
    void loadDefaultFont_( float fontSize, float yOffset = 0.0f );
};

}
