#pragma once
#include "MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_PDF )
#include <filesystem>

#include "hpdf.h"

namespace MR
{

/**
 * @brief Parameters of document style
 */
struct PdfParameters
{
    HPDF_REAL titleSize = 18.f;
    HPDF_REAL textSize = 14.f;
    /**
     * @brief Font name
     * list of available fonts:
     * Courier (-Bold, -Oblique, -BoldOblique)
     * Helvetica (-Bold, -Oblique, -BoldOblique)
     * Times (-Roman, -Bold, -Italic, -BoldItalic)
     * Symbol
     * ZapfDingbats
     */
    std::string fontName = "Helvetica";
};

/**
 * Class for simple creation pdf.
 */
class Pdf
{
public:
    /// Ctor
    MRMESH_API Pdf( const std::filesystem::path& documentPath, const PdfParameters& params = PdfParameters() );
    /// Dtor. Automatically do close
    MRMESH_API ~Pdf();

    Pdf( const Pdf& rhs ) = delete;
    Pdf& operator = ( const Pdf& rhs ) = delete;

    /**
     * Add text block in current cursor position.
     * Move cursor.
     * Box horizontal size is page width without offset.
     * Box vertical size is automatically for text.
     * horAlignment = left
     * if isTitle - horAlignment = center, use titleFontSize
     */
    MRMESH_API void addText( const std::string& text, bool isTitle = false );

    /**
     * @brief Add image from file in current cursor position.
     * If image bigger than page size, autoscale image to page size.
     * Move cursor.
     * @param valuesMarks if not empty - add marks under image.
     * valuesMarks contains pairs<relative_position, marks_text>. 
     *     relative_position is in range [0., 1.], where 0. - left border of image, 1. - right border
     * @param caption if not empty - add caption under marks (if exist) or image.
     */
    MRMESH_API void addImageFromFile( const std::filesystem::path& imagePath, const std::string& caption = {},
        const std::vector<std::pair<double, std::string>>& valuesMarks = {} );

    /// Add new pageand move cursor on it
    MRMESH_API void newPage();

    /// Save and close document. After this impossible add anything in document
    MRMESH_API void close();

    void setCursorPosX( HPDF_REAL posX ) { cursorX_ = posX; };
    void setCursorPosY( HPDF_REAL posY ) { cursorY_ = posY; };
    float getCursorPosX() const { return cursorX_; };
    float getCursorPosY() const { return cursorY_; };

    /// Checking the ability to work with a document
    operator bool() const { return document_ != 0; };

private:
    HPDF_Doc document_ = nullptr;
    HPDF_Page activePage_ = nullptr;
    HPDF_Font activeFont_ = nullptr;

    const std::filesystem::path filename_;

    PdfParameters params_;

    HPDF_REAL cursorX_ = 0;
    HPDF_REAL cursorY_ = 0;

    bool checkDocument() const { return document_ && activePage_; };
};

}
#endif
