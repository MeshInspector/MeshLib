#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_PDF
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRColor.h"

#include <filesystem>
#include <vector>

namespace MR
{

/**
 * @brief Parameters of document style
 */
struct PdfParameters
{
    float titleSize = 18.f;
    float textSize = 14.f;
    /**
     * @brief Font name
     * list of available fonts:
     * Courier (-Bold, -Oblique, -BoldOblique)
     * Helvetica (-Bold, -Oblique, -BoldOblique)
     * Times (-Roman, -Bold, -Italic, -BoldItalic)
     * Symbol
     * ZapfDingbats
     */
    std::string defaultFontName = "Helvetica";
    /**
    * Font name for table (monospaced)
    */
    std::string tableFontName = "Courier";
};

/**
 * Class for simple creation pdf.
 */
class Pdf
{
public:
    /// Ctor
    MRIOEXTRAS_API Pdf( const std::filesystem::path& documentPath, const PdfParameters& params = PdfParameters() );
    MRIOEXTRAS_API Pdf( Pdf&& other ) noexcept;
    MRIOEXTRAS_API Pdf& operator=( Pdf other ) noexcept; // Sic, passing by value.
    /// Dtor. Automatically do close
    MRIOEXTRAS_API ~Pdf();

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
    MRIOEXTRAS_API void addText( const std::string& text, bool isTitle = false );

    /**
     * Add set of pair string - value in current cursor position.
     * Move cursor.
     * Box horizontal size is page width without offset.
     * Box vertical size is automatically for text.
     */
    MRIOEXTRAS_API void addTable( const std::vector<std::pair<std::string, float>>& table );

    struct PaletteRowStats
    {
        Color color;
        std::string rangeMin;
        std::string rangeMax;
        std::string percent;
    };
    MRIOEXTRAS_API void addPaletteStatsTable( const std::vector<PaletteRowStats>& paletteStats );

    /// Parameters to adding image from file
    struct ImageParams
    {
        /// image size in page space
        /// if == {0, 0} - use image size
        /// if .x or .y < 0 use the available page size from the current cursor position (caption size is also accounted for)
        Vector2f size;
        /// caption if not empty - add caption under marks (if exist) or image.
        std::string caption;
        /// set height to keep same scale as width scale
        bool uniformScaleFromWidth = false;
    };
    /**
     * @brief Add image from file in current cursor position.
     * If image bigger than page size, autoscale image to page size.
     * Move cursor.
     */
    MRIOEXTRAS_API void addImageFromFile( const std::filesystem::path& imagePath, const ImageParams& params );

    /// Add new pageand move cursor on it
    MRIOEXTRAS_API void newPage();

    /// Save and close document. After this impossible add anything in document
    MRIOEXTRAS_API void close();

    void setCursorPosX( float posX ) { cursorX_ = posX; }
    void setCursorPosY( float posY ) { cursorY_ = posY; }
    float getCursorPosX() const { return cursorX_; }
    float getCursorPosY() const { return cursorY_; }

    MRIOEXTRAS_API Vector2f getPageSize() const;
    MRIOEXTRAS_API Box2f getPageWorkArea() const;

    /// Checking the ability to work with a document
    MRIOEXTRAS_API operator bool() const;

private:
    struct TextParams;
    // common method for adding different types of text
    void addText_( const std::string& text, const TextParams& params );

    struct State;
    std::unique_ptr<State> state_;

    std::filesystem::path filename_;

    PdfParameters params_;

    float cursorX_ = 0;
    float cursorY_ = 0;

    bool checkDocument() const;
    void moveCursorToNewLine();
};

}
#endif
