#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_PDF
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRExpected.h"

#include <filesystem>
#include <vector>
#include <variant>
#include <optional>

namespace MR
{

/// Fonts included in libharu
/// please note that using default font does not allow UTF-8 encoding
enum class PdfBuildinFont
{
    Courier,
    CourierBold,
    CourierOblique,
    CourierBoldOblique,
    Helvetica,
    HelveticaBold,
    HelveticaOblique,
    HelveticaBoldOblique,
    TimesRoman,
    TimesBold,
    TimesItalic,
    TimesBoldItalic,
    Symbol,
    ZapfDingbats,
    Count
};

using PdfGeneralFont = std::variant<PdfBuildinFont, std::filesystem::path>;

/**
 * @brief Parameters of document style
 */
struct PdfParameters
{
    float titleSize = 18.f;
    float textSize = 14.f;

    /**
     * @brief Font name
     */
    PdfGeneralFont defaultFont = PdfBuildinFont::Helvetica;
    PdfGeneralFont defaultFontBold = PdfBuildinFont::HelveticaBold;
    /**
    * Font name for table (monospaced)
    */
    PdfGeneralFont tableFont = PdfBuildinFont::Courier;
    PdfGeneralFont tableFontBold = PdfBuildinFont::CourierBold;
};

/**
 * Class for simple creation pdf.
 */
class Pdf
{
public:
    /// Ctor. Create a document, but not a page. To create a new page use newPage() method
    MRIOEXTRAS_API Pdf( const PdfParameters& params = PdfParameters() );
    MRIOEXTRAS_API Pdf( Pdf&& other ) noexcept;
    MRIOEXTRAS_API Pdf& operator=( Pdf other ) noexcept; // Sic, passing by value.
    /// Dtor.
    MRIOEXTRAS_API ~Pdf();

    enum class AlignmentHorizontal
    {
        Left,
        Center,
        Right
    };

    // parameters to drawing text
    struct TextParams
    {
        PdfGeneralFont fontName = PdfBuildinFont::Helvetica;

        float fontSize = 14.f;

        AlignmentHorizontal alignment = AlignmentHorizontal::Left;
        Color colorText = Color::black();
        bool underline = false;
    };
    /**
     * Add text block in current cursor position.
     * Move cursor.
     * Box horizontal size is page width without offset.
     * Box vertical size is automatically for text.
     * horAlignment = left
     * if isTitle - horAlignment = center, use titleFontSize
     */
    MRIOEXTRAS_API void addText( const std::string& text, bool isTitle = false );
    MRIOEXTRAS_API void addText( const std::string& text, const TextParams& params );
    /// return text width
    MRIOEXTRAS_API float getTextWidth( const std::string& text, const TextParams& params );

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
        enum class UniformScale
        {
            None,
            FromWidth,
            FromHeight
        } uniformScale = UniformScale::None;
        enum class AlignmentVertical
        {
            Top,
            Center,
            Bottom
        } alignmentVertical = AlignmentVertical::Top;
        AlignmentHorizontal alignmentHorizontal = AlignmentHorizontal::Left;
    };
    /**
     * @brief Add image from file in current cursor position.
     * If image bigger than page size, autoscale image to page size.
     * Move cursor.
     */
    MRIOEXTRAS_API void addImageFromFile( const std::filesystem::path& imagePath, const ImageParams& params );

    /// Add new pageand move cursor on it
    MRIOEXTRAS_API void newPage();
    /// set function to customize new page after creation
    void setNewPageAction( std::function<void(Pdf&)> action ) { newPageAction_ =  action; }

    /// Save document to file
    MRIOEXTRAS_API void saveToFile( const std::filesystem::path& documentPath );

    void setCursorPosX( float posX ) { cursorX_ = posX; }
    void setCursorPosY( float posY ) { cursorY_ = posY; }
    float getCursorPosX() const { return cursorX_; }
    float getCursorPosY() const { return cursorY_; }

    MRIOEXTRAS_API Vector2f getPageSize() const;
    MRIOEXTRAS_API Box2f getPageWorkArea() const;

    /// Checking the ability to work with a document
    MRIOEXTRAS_API operator bool() const;
    

    // Table part
    struct EmptyCell {};
    // class to convert values to string with set format
    struct Cell {
        using Value = std::variant<int, float, bool, std::string, EmptyCell>;
        Value data;

        template<typename T>
        Cell( T value ) : data( value ) {}

        // get strang from contained value
        // \param fmtStr format string like fmt::format
        MRIOEXTRAS_API std::string toString( const std::string& fmtStr = "{}" ) const;
    };
    // set up new table (clear table customization, reset parameters to default values)
    MRIOEXTRAS_API void newTable( int columnCount );
    // set table column widths
    MRIOEXTRAS_API Expected<void> setTableColumnWidths( const std::vector<float>& widths );
    // add in pdf table row with titles
    MRIOEXTRAS_API Expected<void> addTableTitles( const std::vector<std::string>& titles );
    // set format for conversion values to string for each column
    MRIOEXTRAS_API Expected<void> setColumnValuesFormat( const std::vector<std::string>& formats );
    // add in pdf table row with values
    MRIOEXTRAS_API Expected<void> addRow( const std::vector<Cell>& cells );
    // parameters to customization table cell
    // return text width (for table font parameters)
    MRIOEXTRAS_API float getTableTextWidth( const std::string& text );
    struct CellCustomParams
    {
        std::optional<Color> colorText;
        std::optional<Color> colorCellBg;
        std::optional<Color> colorCellBorder;
        std::optional<std::string> text;
    };
    using TableCustomRule = std::function<CellCustomParams( int row, int column, const std::string& cellValueText)>;
    // add rule to customize table cells
    void setTableCustomRule( TableCustomRule rule ) { tableCustomRule_ = rule; }

    // basic drawing methods without automatic cursor position control

    // draw text in specific rect on page
    // text will be cropped by rect
    MRIOEXTRAS_API void drawTextInRect( const std::string& text, const Box2f& rect, const TextParams& params );

    struct TextCellParams
    {
        TextParams textParams;

        Box2f rect;
        Color colorBorder = Color::transparent();
        Color colorBackground = Color::transparent();
    };
    MRIOEXTRAS_API void drawTextCell( const std::string& text, const TextCellParams& params );

private:
    // draw rect (filled with border)
    MRIOEXTRAS_API void drawRect_( const Box2f& rect, const Color& fillColor, const Color& strokeColor );

    // close pdf document without saving. After this impossible add anything in document.
    void reset_();

    // calculate the width of the lines, taking into account automatic hyphenation
    std::vector<float> calcTextLineWidths_( const std::string& text, float width, const TextParams& params );

    struct State;
    std::unique_ptr<State> state_;

    PdfParameters params_;

    std::function<void( Pdf& )> newPageAction_;

    float cursorX_ = 0;
    float cursorY_ = 0;

    bool checkDocument_( const std::string& logAction );
    void moveCursorToNewLine();

    // table parts
    int rowCounter_ = 0;
    struct ColumnInfo
    {
        float width = 100;
        std::string valueFormat = "{}";
    };
    std::vector<ColumnInfo> columnsInfo_;
    TableCustomRule tableCustomRule_;
    struct TableGeneralParams
    {
        Color colorTitleText = Color::white();
        Color colorTitleBg{ 42, 102, 246 };

        Color colorCellText = Color::black();
        Color colorCellBg1{ 170, 194, 251 };
        Color colorCellBg2{ 212, 224, 253 };

        Color colorLines = Color::white();

        float fontSize = 12.f;
    } tableParams_;
};

}
#endif
