#pragma once
#ifndef __EMSCRIPTEN__
#include "MRMesh/MRMeshFwd.h"
#include <filesystem>

namespace PoDoFo
{
class PdfStreamedDocument;
class PdfPainter;
class PdfPage;
class PdfFont;
class PdfImage;
}

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
     * @detail list of available fonts:
     * Courier (-Bold, -Oblique, -BoldOblique)
     * Helvetica (-Bold, -Oblique, -BoldOblique)
     * Times (-Roman, -Italic, -BoldItalic)
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
    /// Horizontal alignment of object in box
    enum class HorAlignment
    {
        Left,
        Center,
        Right
    };

    /// Vertical alignment of object in box
    enum class VertAlignment
    {
        Top,
        Center,
        Bottom
    };

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
     * Direct adding text.
     * Don't move cursor.
     * Manually set box position & size and alignment.
     * Uses text size.
     */
    MRMESH_API void addTextManual( const std::string& text, const Box2d& box, HorAlignment horAlignment, VertAlignment vertAlignment);

    /**
     * @brief Add image from file in current cursor position.
     * @detail If image bigger than page size, autoscale image to page size.
     * Move cursor.
     * @param valuesMarks if not empty - add marks under image.
     * valuesMarks contains pairs<relative_position, marks_text>. 
     *     relative_position is in range [0., 1.], where 0. - left border of image, 1. - right border
     * @param caption if not empty - add caption under marks (if exist) or image.
     */
    MRMESH_API void addImageFromFile( const std::filesystem::path& imagePath, const std::string& caption = std::string(),
        const std::vector<std::pair<double, std::string>>& valuesMarks = {} );

    /**
     * Direct adding image from file.
     * Don't move cursor.
     * Manually set box position & size and alignment.
     * Autoscale image to box size.
     */
    MRMESH_API void addImageFromFileManual( const std::filesystem::path& imagePath, const Box2d& box,
        HorAlignment horAlignment = HorAlignment::Center, VertAlignment vertAlignment  = VertAlignment::Center);
    
    /// Add new pageand move cursor on it
    MRMESH_API void newPage();

    /// Save and close document. After this impossible add anything in document
    MRMESH_API void close();

    void setCursorPosX( double posX ) { cursorX_ = posX; };
    void setCursorPosY( double posY ) { cursorY_ = posY; };
    double getCursorPosX() const { return cursorX_; };
    double getCursorPosY() const { return cursorY_; };

    /// Checking the ability to work with a document
    operator bool() const { return document_ != 0; };

private:
    std::unique_ptr<PoDoFo::PdfStreamedDocument> document_;
    std::unique_ptr<PoDoFo::PdfPainter> painter_;
    PoDoFo::PdfPage* activePage_ = nullptr;
    PoDoFo::PdfFont* activeFont_ = nullptr;

    PdfParameters params_;

    double cursorX_ = 0;
    double cursorY_ = 0;

    bool checkDocument() const { return document_ && painter_ && activePage_; };
};

}
#endif
