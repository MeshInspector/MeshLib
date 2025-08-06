#include "MRPdf.h"
#ifndef MRIOEXTRAS_NO_PDF
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRMesh/MRImageSave.h"

#include <fstream>
#include <vector>

#undef NOMINMAX

#include <hpdf.h>

namespace MR
{

namespace
{
// size of A4 page in pixels (uses 72 PPI)
// https://www.papersizes.org/a-sizes-in-pixels.htm
// TODO need get this value from PoDoFo
constexpr HPDF_REAL pageWidth = 595.;
constexpr HPDF_REAL pageHeight = 842.;
constexpr HPDF_REAL scaleFactor = static_cast<HPDF_REAL>(17. / 6.); // ~2.8(3)

constexpr HPDF_REAL borderFieldLeft = 20 * scaleFactor;
constexpr HPDF_REAL borderFieldRight = pageWidth - 10 * scaleFactor;
constexpr HPDF_REAL borderFieldTop = pageHeight - 10 * scaleFactor;
constexpr HPDF_REAL borderFieldBottom = 10 * scaleFactor;
constexpr HPDF_REAL pageWorkWidth = borderFieldRight - borderFieldLeft;
//constexpr HPDF_REAL pageWorkHeight = borderFieldTop - borderFieldBottom;

constexpr HPDF_REAL spacing = 6 * scaleFactor;

constexpr HPDF_REAL textSpacing = 4 * scaleFactor;
constexpr HPDF_REAL lineSpacingScale = 1.2f;

constexpr HPDF_REAL labelHeight = 10 * scaleFactor;

constexpr float tableCellPaddingX = 4 * scaleFactor;
constexpr float tableCellPaddingY = 1 * scaleFactor;

// count the number of rows with auto-transfer in mind for a given page (page, font and font size)
int calcTextLinesCount( HPDF_Doc doc, HPDF_Page page, const std::string& text )
{
    HPDF_REAL r;
    HPDF_UINT substrStart = 0;
    int count = 0;
    for ( ; substrStart < text.size(); ++count )
    {
        HPDF_UINT lineSize = HPDF_Page_MeasureText( page, text.data() + substrStart, pageWorkWidth, HPDF_TRUE, &r );
        if ( lineSize == 0 && HPDF_GetError( doc ) != HPDF_OK )
            break;
        substrStart += lineSize;
    }
    return count;
}

}

struct Pdf::State
{
    HPDF_Doc document = nullptr;
    HPDF_Page activePage = nullptr;
    HPDF_Font defaultFont = nullptr;
    HPDF_Font tableFont = nullptr;
};

struct Pdf::TextParams
{
    HPDF_Font font = nullptr;
    float fontSize = 14.f;
    HPDF_TextAlignment alignment = HPDF_TALIGN_LEFT;
    bool drawBorder = false;
    static TextParams title( const Pdf& pdf )
    {
        return TextParams{ pdf.state_->defaultFont, pdf.params_.titleSize, HPDF_TALIGN_CENTER };
    }
    static TextParams text( const Pdf& pdf )
    {
        return TextParams{ pdf.state_->defaultFont, pdf.params_.textSize, HPDF_TALIGN_LEFT };
    }
    static TextParams table( const Pdf& pdf )
    {
        return TextParams{ pdf.state_->tableFont, pdf.params_.textSize, HPDF_TALIGN_LEFT };
    }
};

Pdf::Pdf( const std::filesystem::path& documentPath, const PdfParameters& params /*= PdfParameters()*/ )
    : state_( std::make_unique<State>() )
    , filename_{ documentPath }
    , params_( params )
{
    cursorX_ = borderFieldLeft;
    cursorY_ = borderFieldTop;

    state_->document = HPDF_New( NULL, NULL );
    if ( !state_->document )
    {
        spdlog::warn( "Can't create PDF document. HPDF error code {}", HPDF_GetError( state_->document ) );
        return;
    }
    HPDF_SetCompressionMode( state_->document, HPDF_COMP_ALL );
    state_->activePage = HPDF_AddPage( state_->document );
    if ( !state_->activePage )
    {
        spdlog::warn( "Can't create page. HPDF error code {}", HPDF_GetError( state_->document ) );
        return;
    }

    HPDF_Page_SetSize( state_->activePage, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT );
    state_->defaultFont = HPDF_GetFont( state_->document, params_.defaultFontName.c_str(), NULL );
    if ( !state_->defaultFont )
    {
        spdlog::debug( "Can't find font: \"{}\". HPDF error code {}", params_.defaultFontName, HPDF_GetError( state_->document ) );
        return;
    }
    state_->tableFont = HPDF_GetFont( state_->document, params_.tableFontName.c_str(), NULL );
    if ( !state_->tableFont )
    {
        spdlog::debug( "Can't find font: \"{}\". HPDF error code {}", params_.tableFontName, HPDF_GetError( state_->document ) );
        return;
    }

    HPDF_Page_SetFontAndSize( state_->activePage, state_->defaultFont, params_.textSize );
}

Pdf::Pdf( Pdf&& other ) noexcept
    : state_( std::move( other.state_ ) )
{
    other.state_ = {};
}

Pdf& Pdf::operator=( Pdf other ) noexcept
{
    std::swap( state_, other.state_ );
    return *this;
}

Pdf::~Pdf()
{
    close();
}

void Pdf::addText(const std::string& text, bool isTitle /*= false*/ )
{
    addText_( text, isTitle ? TextParams::title( *this ) : TextParams::text( *this ) );
}

void Pdf::addTable( const std::vector<std::pair<std::string, float>>& table )
{
    if ( table.empty() )
        return;

    const size_t maxFirstSize = std::max_element( table.begin(), table.end(), [] (const std::pair<std::string, float>& lhv, const std::pair<std::string, float>& rhv)
    {
        return lhv.first.length() < rhv.first.length();
    } )->first.length();

    std::vector<std::string> valueStrs( table.size() );
    for ( int i = 0; i < table.size(); ++i )
        valueStrs[i] = fmt::format( "{:.5f}", table[i].second );
    const size_t maxSecondSize = std::max_element( valueStrs.begin(), valueStrs.end(), [] ( const std::string& lhv, const std::string& rhv )
    {
        return lhv.length() < rhv.length();
    } )->length();

    std::string resStr;
    resStr += fmt::format( "{: <{}} : {: >{}}", table[0].first, maxFirstSize, valueStrs[0], maxSecondSize );
    for ( int i = 1; i < table.size(); ++i )
    {
        resStr += fmt::format( "\n{: <{}} : {: >{}}", table[i].first, maxFirstSize, valueStrs[i], maxSecondSize );
    }
    addText_( resStr, TextParams::table( *this ) );
}

void Pdf::addPaletteStatsTable( const std::vector<PaletteRowStats>& paletteStats )
{
    if ( !state_->document )
    {
        spdlog::warn( "Can't add text to pdf page: no valid document" );
        return;
    }

    HPDF_Page_SetFontAndSize( state_->activePage, state_->tableFont, params_.textSize );

    size_t longestMin = std::max_element( paletteStats.begin(), paletteStats.end(), [] ( const PaletteRowStats& lhv, const PaletteRowStats& rhv )
    {
        return lhv.rangeMin.length() < rhv.rangeMin.length();
    } )->rangeMin.length();
    size_t longestMax = std::max_element( paletteStats.begin(), paletteStats.end(), [] ( const PaletteRowStats& lhv, const PaletteRowStats& rhv )
    {
        return lhv.rangeMax.length() < rhv.rangeMax.length();
    } )->rangeMax.length();
    size_t maxWidth = std::max( std::max( longestMin, longestMax ), size_t( 9 ) ); // "Range Min".length() == 9
    std::string dummy = fmt::format( "{: >{}}", "_", maxWidth );

    const float rangeLimColumnWidth = HPDF_Page_TextWidth( state_->activePage, dummy.c_str() );

    float bordersX[5];
    bordersX[0] = borderFieldLeft;
    bordersX[1] = bordersX[0] + tableCellPaddingX * 4.f + HPDF_Page_TextWidth( state_->activePage, "Color" );
    bordersX[2] = bordersX[1] + tableCellPaddingX * 4.f + rangeLimColumnWidth;
    bordersX[3] = bordersX[2] + tableCellPaddingX * 4.f + rangeLimColumnWidth;
    bordersX[4] = bordersX[3] + tableCellPaddingX * 4.f + HPDF_Page_TextWidth( state_->activePage, "% of All" );


    const auto textHeight = static_cast< HPDF_REAL >( params_.textSize * lineSpacingScale );
    const float cellHeight = textHeight + tableCellPaddingY * 2.f;

    if ( cursorY_ - cellHeight * paletteStats.size() < borderFieldBottom )
        newPage();

    HPDF_Page_SetTextLeading( state_->activePage, params_.textSize * lineSpacingScale );

    auto drawCellBorders = [&] ()
    {
        for ( int i = 0; i < 4; ++i )
        {
            HPDF_Page_Rectangle( state_->activePage, bordersX[i], cursorY_ - cellHeight, bordersX[i + 1] - bordersX[i], cellHeight );
            HPDF_Page_Stroke( state_->activePage );
        }
    };

    drawCellBorders();

    auto drawTextInCell = [&] ( int cellIndex, const std::string& text, bool alignCenter = false )
    {
        HPDF_Page_TextRect( state_->activePage, bordersX[cellIndex] + tableCellPaddingX, cursorY_ - tableCellPaddingY,
            bordersX[cellIndex + 1] - tableCellPaddingX, cursorY_ - textHeight - tableCellPaddingY,
            text.c_str(), alignCenter ? HPDF_TALIGN_CENTER : HPDF_TALIGN_RIGHT, nullptr);
    };

    HPDF_Page_BeginText( state_->activePage );
    drawTextInCell( 0, "Color", true );
    drawTextInCell( 1, "Range Min", true );
    drawTextInCell( 2, "Range Max", true );
    drawTextInCell( 3, "% of All", true );
    HPDF_Page_EndText( state_->activePage );
    cursorY_ -= cellHeight;

    UniqueTemporaryFolder pathFolder( {} );
    std::filesystem::path imageCellPath = pathFolder / "pdf_palette_cell.png";
    Image mrImage;
    mrImage.resolution = { int( std::floor( bordersX[1] - bordersX[0] ) ), int( cellHeight ) };
    for ( int i = 0; i < paletteStats.size(); ++i )
    {
        drawCellBorders();

        mrImage.pixels = std::vector<Color>( mrImage.resolution.x * mrImage.resolution.y, paletteStats[i].color );
        std::ignore = ImageSave::toAnySupportedFormat( mrImage, imageCellPath );
        HPDF_Image pdfImage = HPDF_LoadPngImageFromFile( state_->document, utf8string( imageCellPath ).c_str() ); // TODO FIX need rework without using filesystem
        if ( pdfImage )
            HPDF_Page_DrawImage( state_->activePage, pdfImage, bordersX[0], cursorY_ - cellHeight, std::floor( bordersX[1] - bordersX[0] ), cellHeight );
        
        HPDF_Page_BeginText( state_->activePage );
        drawTextInCell( 1, paletteStats[i].rangeMin );
        drawTextInCell( 2, paletteStats[i].rangeMax );
        drawTextInCell( 3, paletteStats[i].percent );
        HPDF_Page_EndText( state_->activePage );
        cursorY_ -= cellHeight;
    }

    moveCursorToNewLine();
}


void Pdf::addImageFromFile( const std::filesystem::path& imagePath, const ImageParams& params )
{
    if ( !state_->document )
    {
        spdlog::warn( "Can't add image to pdf page: no valid document" );
        return;
    }

    HPDF_Image pdfImage = HPDF_LoadPngImageFromFile( state_->document, utf8string( imagePath ).c_str() );
    if ( !pdfImage )
    {
        spdlog::warn( "Failed to load image from file. HPDF error code {}", HPDF_GetError( state_->document ) );
        return;
    }

    const HPDF_REAL additionalHeight = labelHeight * !params.caption.empty();
    HPDF_REAL imageWidth = params.size.x;
    if ( imageWidth == 0.f )
        imageWidth = (HPDF_REAL) HPDF_Image_GetWidth( pdfImage );
    else if ( imageWidth < 0.f )
        imageWidth = borderFieldRight - cursorX_;
    HPDF_REAL imageHeight = params.size.y;
    if ( params.uniformScaleFromWidth )
        imageHeight = imageWidth * HPDF_Image_GetHeight( pdfImage ) / HPDF_Image_GetWidth( pdfImage );
    else if ( imageHeight == 0.f )
        imageHeight = (HPDF_REAL) HPDF_Image_GetHeight( pdfImage );
    else if ( imageHeight < 0.f )
        imageHeight = cursorY_ - borderFieldBottom - additionalHeight;

    if ( cursorY_ - imageHeight - additionalHeight < borderFieldBottom )
        newPage();

    cursorY_ -= imageHeight;
    HPDF_Page_DrawImage( state_->activePage, pdfImage, cursorX_, cursorY_, imageWidth, imageHeight );

    if ( !params.caption.empty() )
    {
        cursorY_ -= textSpacing / 2.;
        HPDF_Page_BeginText( state_->activePage );
        HPDF_Page_SetFontAndSize( state_->activePage, state_->defaultFont, params_.textSize );
        HPDF_Page_TextRect( state_->activePage, cursorX_, cursorY_, cursorX_ + imageWidth, cursorY_ - labelHeight, params.caption.c_str(), HPDF_TALIGN_CENTER, nullptr );
        HPDF_Page_EndText( state_->activePage );
        cursorY_ -= labelHeight;
    }

    moveCursorToNewLine();
}

void Pdf::newPage()
{
    if ( !state_->document )
    {
        spdlog::warn( "Can't create new pdf page: no valid document" );
        return;
    }

    state_->activePage = HPDF_AddPage( state_->document );
    if ( !state_->activePage )
    {
        spdlog::warn( "Error while creating new pdf page: {}", HPDF_GetError( state_->document ) );
        return;
    }

    cursorX_ = borderFieldLeft;
    cursorY_ = borderFieldTop;
    HPDF_Page_SetSize( state_->activePage, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT);
}

void Pdf::close()
{
    if ( state_->document )
    {
        // reset all errors before saving document
        HPDF_ResetError( state_->document );

        auto pathString = utf8string( filename_ );
        HPDF_SaveToFile( state_->document, pathString.c_str() );

        HPDF_STATUS status = HPDF_GetError( state_->document );
        if (status != HPDF_OK)
        {
            spdlog::error( "HPDF Error while saving pdf: {}", status );
            HPDF_ResetError( state_->document );
        }
        HPDF_Free( state_->document );
        state_->document = nullptr;
    }
    state_->activePage = nullptr;
    state_->defaultFont = nullptr;
}

Vector2f Pdf::getPageSize() const
{
    return { pageWidth, pageHeight };
}

Box2f Pdf::getPageWorkArea() const
{
    return { { borderFieldLeft, borderFieldBottom }, { borderFieldRight, borderFieldTop } };
}

Pdf::operator bool() const
{
    return state_->document != 0;
}

void Pdf::addText_( const std::string& text, const TextParams& textParams )
{
    if ( text.empty() )
        return;

    if ( !state_->document )
    {
        spdlog::warn( "Can't add text to pdf page: no valid document" );
        return;
    }

    HPDF_Page_SetFontAndSize( state_->activePage, textParams.font, textParams.fontSize );

    int strNum = calcTextLinesCount( state_->document, state_->activePage, text );
    const auto textHeight = static_cast< HPDF_REAL >( textParams.fontSize * strNum * lineSpacingScale );

    // need add the ability to transfer text between pages
    if ( cursorY_ - textHeight < borderFieldBottom )
        newPage();

    HPDF_Page_BeginText( state_->activePage );
    HPDF_Page_SetTextLeading( state_->activePage, textParams.fontSize * lineSpacingScale );

    HPDF_Page_TextRect( state_->activePage, cursorX_, cursorY_, cursorX_ + pageWorkWidth, cursorY_ - textHeight, text.c_str(), textParams.alignment, nullptr );
    HPDF_Page_EndText( state_->activePage );

    if ( textParams.drawBorder )
    {
        HPDF_Page_Rectangle( state_->activePage, cursorX_, cursorY_ - textHeight, pageWorkWidth, textHeight );
        HPDF_Page_Stroke( state_->activePage );
    }

    cursorY_ -= textHeight;
    moveCursorToNewLine();
}

bool Pdf::checkDocument() const
{
    return state_->document && state_->activePage;
}

void Pdf::moveCursorToNewLine()
{
    if ( cursorY_ - spacing < borderFieldBottom )
        newPage();
    else
    {
        cursorX_ = borderFieldLeft;
        cursorY_ -= spacing;
    }
}

}
#endif
