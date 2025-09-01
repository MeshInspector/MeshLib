#include "MRPdf.h"
#ifndef MRIOEXTRAS_NO_PDF
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRImage.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRUniqueTemporaryFolder.h"
#include "MRMesh/MRImageSave.h"
#include "MRPch/MRFmt.h"


#include <fstream>
#include <vector>

#undef NOMINMAX

#include <hpdf.h>

namespace
{

std::string GetHpdfErrorDescription( HPDF_STATUS errorCode );

void pdfPrintError( const char* funcName, HPDF_Doc doc, HPDF_STATUS status )
{
    if ( status == HPDF_OK )
        return;

    spdlog::warn( "Pdf: Error in {} call.", funcName );
    spdlog::warn( "Failed with error: {} {}", status, GetHpdfErrorDescription( status ) );
    if ( doc )
    {
        status = HPDF_GetErrorDetail( doc );
        if ( status != HPDF_OK )
            spdlog::warn( "Error detail: {} {}", status, std::strerror( status ) );
    }
}

#define MR_HPDF_CHECK_RES_STATUS( ... ) \
do { \
    HPDF_STATUS status = __VA_ARGS__; \
    if (status != HPDF_OK) { \
        pdfPrintError( #__VA_ARGS__, state_->document, status ); \
    } \
} while (0)

#define MR_HPDF_CHECK_ERROR( ... ) \
([&]() -> decltype(auto) { \
    decltype(auto) result = __VA_ARGS__; \
    HPDF_STATUS status = HPDF_GetError( state_->document ); \
    if (status != HPDF_OK) { \
        pdfPrintError( #__VA_ARGS__, state_->document, status ); \
    } \
    return result; \
})()

// error text getting from https://github.com/libharu/libharu/wiki/Error-handling
std::string GetHpdfErrorDescription( HPDF_STATUS errorCode )
{
    switch ( errorCode )
    {
    case 0x1001: return "Internal error. Data consistency was lost.";
    case 0x1002: return "Internal error. Data consistency was lost.";
    case 0x1003: return "Internal error. Data consistency was lost.";
    case 0x1004: return "Data length > HPDF_LIMIT_MAX_STRING_LEN.";
    case 0x1005: return "Cannot get pallet data from PNG image.";
    case 0x1007: return "Dictionary elements > HPDF_LIMIT_MAX_DICT_ELEMENT";
    case 0x1008: return "Internal error. Data consistency was lost.";
    case 0x1009: return "Internal error. Data consistency was lost.";
    case 0x100A: return "Internal error. Data consistency was lost.";
    case 0x100B: return "HPDF_SetEncryptMode() or HPDF_SetPermission() called before password set.";
    case 0x100C: return "Internal error. Data consistency was lost.";
    case 0x100E: return "Tried to re-register a registered font.";
    case 0x100F: return "Cannot register a character to the Japanese word wrap characters list.";
    case 0x1011: return "Tried to set the owner password to NULL or owner/user passwords are the same.";
    case 0x1013: return "Internal error. Data consistency was lost.";
    case 0x1014: return "Stack depth > HPDF_LIMIT_MAX_GSTATE.";
    case 0x1015: return "Memory allocation failed.";
    case 0x1016: return "File processing failed. (Detailed code is set.)";
    case 0x1017: return "Cannot open a file. (Detailed code is set.)";
    case 0x1019: return "Tried to load a font that has been registered.";
    case 0x101A: return "Font-file format is invalid or internal error.";
    case 0x101B: return "Cannot recognize header of afm file.";
    case 0x101C: return "Specified annotation handle is invalid.";
    case 0x101E: return "Bit-per-component of a mask-image is invalid.";
    case 0x101F: return "Cannot recognize char-matrics-data of afm file.";
    case 0x1020: return "Invalid color space or usage.";
    case 0x1021: return "Invalid value set when invoking HPDF_SetCommpressionMode().";
    case 0x1022: return "An invalid date-time value was set.";
    case 0x1023: return "An invalid destination handle was set.";
    case 0x1025: return "An invalid document handle was set.";
    case 0x1026: return "Function invalid in the present state.";
    case 0x1027: return "An invalid encoder handle was set.";
    case 0x1028: return "Combination between font and encoder is wrong.";
    case 0x102B: return "An invalid encoding name is specified.";
    case 0x102C: return "Encryption key length is invalid.";
    case 0x102D: return "Invalid font handle or unsupported font format.";
    case 0x102E: return "Internal error. Data consistency was lost.";
    case 0x102F: return "Font with the specified name is not found.";
    case 0x1030: return "Unsupported image format.";
    case 0x1031: return "Unsupported JPEG image format.";
    case 0x1032: return "Cannot read a postscript-name from an afm file.";
    case 0x1033: return "Invalid object or internal error.";
    case 0x1034: return "Internal error. Data consistency was lost.";
    case 0x1035: return "Invalid image mask operation.";
    case 0x1036: return "An invalid outline handle was specified.";
    case 0x1037: return "An invalid page handle was specified.";
    case 0x1038: return "An invalid pages handle was specified.";
    case 0x1039: return "An invalid value is set.";
    case 0x103B: return "Invalid PNG image format.";
    case 0x103C: return "Internal error. Data consistency was lost.";
    case 0x103D: return "Missing _FILE_NAME entry for delayed loading.";
    case 0x103F: return "Invalid .TTC file format.";
    case 0x1040: return "TTC index > number of fonts.";
    case 0x1041: return "Cannot read width-data from afm file.";
    case 0x1042: return "Internal error. Data consistency was lost.";
    case 0x1043: return "Error returned from PNGLIB while loading image.";
    case 0x1044: return "Internal error. Data consistency was lost.";
    case 0x1045: return "Internal error. Data consistency was lost.";
    case 0x1049: return "Internal error. Data consistency was lost.";
    case 0x104A: return "Internal error. Data consistency was lost.";
    case 0x104B: return "Internal error. Data consistency was lost.";
    case 0x104C: return "No graphics-states to be restored.";
    case 0x104D: return "Internal error. Data consistency was lost.";
    case 0x104E: return "The current font is not set.";
    case 0x104F: return "An invalid font-handle was specified.";
    case 0x1050: return "An invalid font-size was set.";
    case 0x1051: return "Invalid graphics mode.";
    case 0x1052: return "Internal error. Data consistency was lost.";
    case 0x1053: return "Rotate value is not multiple of 90.";
    case 0x1054: return "Invalid page-size was set.";
    case 0x1055: return "An invalid image handle was set.";
    case 0x1056: return "The specified value is out of range.";
    case 0x1057: return "The specified value is out of range.";
    case 0x1058: return "Unexpected EOF marker detected.";
    case 0x1059: return "Internal error. Data consistency was lost.";
    case 0x105B: return "Text length is too long.";
    case 0x105C: return "Function skipped due to other errors.";
    case 0x105D: return "Font cannot be embedded (license restriction).";
    case 0x105E: return "Unsupported TTF format (unicode cmap not found).";
    case 0x105F: return "Unsupported TTF format.";
    case 0x1060: return "Unsupported TTF format (missing table).";
    case 0x1061: return "Internal error. Data consistency was lost.";
    case 0x1062: return "Unsupported function or internal error.";
    case 0x1063: return "Unsupported JPEG format.";
    case 0x1064: return "Failed to parse .PFB file.";
    case 0x1065: return "Internal error. Data consistency was lost.";
    case 0x1066: return "ZLIB function error.";
    case 0x1067: return "Invalid page index.";
    case 0x1068: return "Invalid URI.";
    case 0x1069: return "An invalid page-layout was set.";
    case 0x1070: return "An invalid page-mode was set.";
    case 0x1071: return "An invalid page-num-style was set.";
    case 0x1072: return "Invalid annotation icon.";
    case 0x1073: return "Invalid border style.";
    case 0x1074: return "Invalid page direction.";
    case 0x1075: return "Invalid font handle.";
    default: return "Unknown error.";
    }
}

}

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

}

std::string Pdf::Cell::toString( const std::string& fmtStr /*= "{}"*/ ) const
{
    return std::visit( [&] ( const auto& val ) -> std::string
    {
        using T = std::decay_t<decltype( val )>;
        if constexpr ( std::is_same_v<T, EmptyCell> )
            return "";
        else
            return fmt::format( runtimeFmt( fmtStr ), val );
    }, data );
}


struct Pdf::State
{
    HPDF_Doc document = nullptr;
    HPDF_Page activePage = nullptr;
    HPDF_Font defaultFont = nullptr;
    HPDF_Font defaultFontBold = nullptr;
    HPDF_Font tableFont = nullptr;
    HPDF_Font tableFontBold = nullptr;
};

struct Pdf::TextParams
{
    HPDF_Font font = nullptr;
    float fontSize = 14.f;
    HPDF_TextAlignment alignment = HPDF_TALIGN_LEFT;
    Color colorText = Color::black();

    static TextParams title( const Pdf& pdf )
    {
        return { .font = pdf.state_->defaultFont, .fontSize = pdf.params_.titleSize, .alignment = HPDF_TALIGN_CENTER };
    }
    static TextParams text( const Pdf& pdf )
    {
        return { .font = pdf.state_->defaultFont, .fontSize = pdf.params_.textSize };
    }
    static TextParams table( const Pdf& pdf )
    {
        return { .font = pdf.state_->tableFont, .fontSize = pdf.params_.textSize };
    }
};

struct Pdf::TextCellParams
{
    TextParams textParams;

    Box2f rect;
    Color colorBorder = Color::transparent();
    Color colorBackground = Color::transparent();

};

Pdf::Pdf( const PdfParameters& params /*= PdfParameters()*/ )
    : state_( std::make_unique<State>() )
    , params_( params )
{
    state_->document = HPDF_New( NULL, NULL );
    if ( !state_->document )
    {
        spdlog::warn( "Pdf: Can't create PDF document." );
        return;
    }
    
    MR_HPDF_CHECK_RES_STATUS( HPDF_SetCompressionMode( state_->document, HPDF_COMP_ALL ) );

    state_->activePage = HPDF_AddPage( state_->document );
    if ( !state_->activePage )
    {
        spdlog::warn( "Pdf: Can't create page." );
        pdfPrintError( "HPDF_AddPage", state_->document, HPDF_GetError( state_->document ) );
        reset_();
        return;
    }

    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetSize( state_->activePage, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT ) );

    state_->defaultFont = HPDF_GetFont( state_->document, params_.defaultFontName.c_str(), NULL );
    if ( !state_->defaultFont )
    {
        spdlog::debug( "Pdf: Can't find font: \"{}\".", params_.defaultFontName );
        pdfPrintError( "HPDF_GetFont", state_->document, HPDF_GetError( state_->document ) );
        return;
    }
    state_->defaultFontBold = HPDF_GetFont( state_->document, params_.defaultFontBoldName.c_str(), NULL );
    if ( !state_->defaultFontBold )
    {
        spdlog::debug( "Pdf: Can't find font: \"{}\".", params_.defaultFontBoldName );
        pdfPrintError( "HPDF_GetFont", state_->document, HPDF_GetError( state_->document ) );
        return;
    }
    state_->tableFont = HPDF_GetFont( state_->document, params_.tableFontName.c_str(), NULL );
    if ( !state_->tableFont )
    {
        spdlog::debug( "Pdf: Can't find font: \"{}\".", params_.tableFontName );
        pdfPrintError( "HPDF_GetFont", state_->document, HPDF_GetError( state_->document ) );
        return;
    }
    state_->tableFontBold = HPDF_GetFont( state_->document, params_.tableFontBoldName.c_str(), NULL );
    if ( !state_->tableFontBold )
    {
        spdlog::debug( "Pdf: Can't find font: \"{}\".", params_.tableFontBoldName );
        pdfPrintError( "HPDF_GetFont", state_->document, HPDF_GetError( state_->document ) );
        return;
    }

    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetFontAndSize( state_->activePage, state_->defaultFont, params_.textSize ) );

    cursorX_ = borderFieldLeft;
    cursorY_ = borderFieldTop;
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
    reset_();
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
        spdlog::warn( "Pdf: Can't add text to pdf page: no valid document" );
        return;
    }

    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetFontAndSize( state_->activePage, state_->tableFont, params_.textSize ) );

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

    const float rangeLimColumnWidth = MR_HPDF_CHECK_ERROR( HPDF_Page_TextWidth( state_->activePage, dummy.c_str() ) );

    float bordersX[5];
    bordersX[0] = borderFieldLeft;
    bordersX[1] = bordersX[0] + tableCellPaddingX * 4.f + MR_HPDF_CHECK_ERROR( HPDF_Page_TextWidth( state_->activePage, "Color" ) );
    bordersX[2] = bordersX[1] + tableCellPaddingX * 4.f + rangeLimColumnWidth;
    bordersX[3] = bordersX[2] + tableCellPaddingX * 4.f + rangeLimColumnWidth;
    bordersX[4] = bordersX[3] + tableCellPaddingX * 4.f + MR_HPDF_CHECK_ERROR( HPDF_Page_TextWidth( state_->activePage, "% of All" ) );


    const auto textHeight = static_cast< HPDF_REAL >( params_.textSize * lineSpacingScale );
    const float cellHeight = textHeight + tableCellPaddingY * 2.f;

    if ( cursorY_ - cellHeight * paletteStats.size() < borderFieldBottom )
        newPage();

    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetTextLeading( state_->activePage, params_.textSize * lineSpacingScale ) );

    auto drawCellBorders = [&] ()
    {
        for ( int i = 0; i < 4; ++i )
        {
            MR_HPDF_CHECK_RES_STATUS( HPDF_Page_Rectangle( state_->activePage, bordersX[i], cursorY_ - cellHeight, bordersX[i + 1] - bordersX[i], cellHeight ) );
            MR_HPDF_CHECK_RES_STATUS( HPDF_Page_Stroke( state_->activePage ) );
        }
    };

    drawCellBorders();

    auto drawTextInCell = [&] ( int cellIndex, const std::string& text, bool alignCenter = false )
    {
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_TextRect( state_->activePage, bordersX[cellIndex] + tableCellPaddingX, cursorY_ - tableCellPaddingY,
            bordersX[cellIndex + 1] - tableCellPaddingX, cursorY_ - textHeight - tableCellPaddingY,
            text.c_str(), alignCenter ? HPDF_TALIGN_CENTER : HPDF_TALIGN_RIGHT, nullptr ) );
    };

    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_BeginText( state_->activePage ) );
    drawTextInCell( 0, "Color", true );
    drawTextInCell( 1, "Range Min", true );
    drawTextInCell( 2, "Range Max", true );
    drawTextInCell( 3, "% of All", true );
    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_EndText( state_->activePage ) );
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
        HPDF_Image pdfImage = MR_HPDF_CHECK_ERROR( HPDF_LoadPngImageFromFile( state_->document, utf8string( imageCellPath ).c_str() ) ); // TODO FIX need rework without using filesystem
        if ( pdfImage )
            MR_HPDF_CHECK_RES_STATUS( HPDF_Page_DrawImage( state_->activePage, pdfImage, bordersX[0], cursorY_ - cellHeight, std::floor( bordersX[1] - bordersX[0] ), cellHeight ) );
        
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_BeginText( state_->activePage ) );
        drawTextInCell( 1, paletteStats[i].rangeMin );
        drawTextInCell( 2, paletteStats[i].rangeMax );
        drawTextInCell( 3, paletteStats[i].percent );
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_EndText( state_->activePage ) );
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

    HPDF_Image pdfImage = MR_HPDF_CHECK_ERROR( HPDF_LoadPngImageFromFile( state_->document, utf8string( imagePath ).c_str() ) );
    if ( !pdfImage )
    {
        spdlog::warn( "Pdf: Failed to load image from file \"{}\"", utf8string( imagePath ) );
        return;
    }

    const HPDF_REAL additionalHeight = labelHeight * !params.caption.empty();
    HPDF_REAL imageWidth = params.size.x;
    if ( imageWidth == 0.f )
        imageWidth = (HPDF_REAL)MR_HPDF_CHECK_ERROR( HPDF_Image_GetWidth( pdfImage ) );
    else if ( imageWidth < 0.f )
        imageWidth = borderFieldRight - cursorX_;
    HPDF_REAL imageHeight = params.size.y;
    if ( params.uniformScaleFromWidth )
        imageHeight = imageWidth * MR_HPDF_CHECK_ERROR( HPDF_Image_GetHeight( pdfImage ) ) / MR_HPDF_CHECK_ERROR( HPDF_Image_GetWidth( pdfImage ) );
    else if ( imageHeight == 0.f )
        imageHeight = (HPDF_REAL)MR_HPDF_CHECK_ERROR( HPDF_Image_GetHeight( pdfImage ) );
    else if ( imageHeight < 0.f )
        imageHeight = cursorY_ - borderFieldBottom - additionalHeight;

    if ( cursorY_ - imageHeight - additionalHeight < borderFieldBottom )
        newPage();

    cursorY_ -= imageHeight;
    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_DrawImage( state_->activePage, pdfImage, cursorX_, cursorY_, imageWidth, imageHeight ) );

    if ( !params.caption.empty() )
    {
        cursorY_ -= textSpacing / 2.;
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_BeginText( state_->activePage ) );
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetFontAndSize( state_->activePage, state_->defaultFont, params_.textSize ) );
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_TextRect( state_->activePage, cursorX_, cursorY_, cursorX_ + imageWidth, cursorY_ - labelHeight, params.caption.c_str(), HPDF_TALIGN_CENTER, nullptr ) );
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_EndText( state_->activePage ) );
        cursorY_ -= labelHeight;
    }

    moveCursorToNewLine();
}

void Pdf::newPage()
{
    if ( !state_->document )
    {
        spdlog::warn( "Pdf: Can't create new pdf page: no valid document" );
        return;
    }

    state_->activePage = HPDF_AddPage( state_->document );
    if ( !state_->activePage )
    {
        spdlog::warn( "Pdf: Error while creating new pdf page: {}", HPDF_GetError( state_->document ) );
        return;
    }

    cursorX_ = borderFieldLeft;
    cursorY_ = borderFieldTop;
    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetSize( state_->activePage, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT) );
}

void Pdf::saveToFile( const std::filesystem::path& documentPath )
{
    if ( documentPath.empty() )
    {
        spdlog::error( "Pdf: Error: empty path." );
        return;
    }

    if ( !state_->document )
    {
        spdlog::warn( "Pdf: Can't save to file: no valid document." );
        return;
    }

    // reset all errors before saving document
    HPDF_ResetError( state_->document );

    auto pathString = utf8string( documentPath );
    HPDF_SaveToFile( state_->document, pathString.c_str() );

    HPDF_STATUS status = HPDF_GetError( state_->document );
    if (status != HPDF_OK)
    {
        spdlog::error( "Pdf: Error while saving pdf to file \"{}\": {}", pathString, status );
        HPDF_ResetError( state_->document );
    }
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

void Pdf::newTable( int columnCount )
{
    if ( columnCount < 1 )
        return;
    columnsInfo_.clear();
    columnsInfo_.resize( columnCount, { pageWorkWidth / columnCount, "{}" } );
    rowCounter_ = 0;
}

Expected<void> Pdf::setTableColumnWidths( const std::vector<float>& widths )
{
    assert( widths.size() == columnsInfo_.size() );
    if ( widths.size() != columnsInfo_.size() )
    {
        return unexpected( "Pdf: mismatch number of columns and widths" );
    }

    for ( int i = 0; i < columnsInfo_.size(); ++i )
        columnsInfo_[i].width = widths[i];
    return {};
}

Expected<void> Pdf::addTableTitles( const std::vector<std::string>& titles )
{
    assert( titles.size() == columnsInfo_.size() );
    if ( titles.size() != columnsInfo_.size() )
    {
        return unexpected( "Pdf: mismatch number of columns and titles" );
    }

    TextCellParams params;
    params.colorBackground = tableParams_.colorTitleBg;
    params.colorBorder = tableParams_.colorLines;
    params.textParams.font = state_->tableFontBold;
    params.textParams.fontSize = tableParams_.fontSize;
    params.textParams.alignment = HPDF_TALIGN_CENTER;
    params.textParams.colorText = tableParams_.colorTitleText;

    const auto textHeight = static_cast< HPDF_REAL >( params.textParams.fontSize ) * 1.6f;
    
    if ( cursorY_ - textHeight < borderFieldBottom )
        newPage();

    float posX = borderFieldLeft;
    for ( int i = 0; i < titles.size(); ++i )
    {
        params.rect = Box2f( { posX, cursorY_ - textHeight }, { posX + columnsInfo_[i].width, cursorY_ } );
        drawTextCell_( titles[i], params );
        posX += columnsInfo_[i].width;
    }
    cursorY_ -= textHeight;
    return {};
}

Expected<void> Pdf::setColumnValuesFormat( const std::vector<std::string>& formats )
{
    if ( formats.size() != columnsInfo_.size() )
        return unexpected( "Pdf: Error set up table column value formats: wrong parameters count." );
    for ( int i = 0; i < columnsInfo_.size(); ++i )
        columnsInfo_[i].valueFormat = formats[i];
    return {};
}

Expected<void> Pdf::addRow( const std::vector<Cell>& cells )
{
    if ( cells.size() != columnsInfo_.size() )
    {
        return unexpected( "Pdf: Error adding table row: wrong parameters count." );
    }

    TextCellParams params;
    params.colorBackground = ( rowCounter_ & 1 ) ? tableParams_.colorCellBg1 : tableParams_.colorCellBg2;
    params.colorBorder = tableParams_.colorLines;
    params.textParams.font = state_->tableFont;
    params.textParams.fontSize = tableParams_.fontSize;
    params.textParams.alignment = HPDF_TALIGN_CENTER;
    params.textParams.colorText = tableParams_.colorCellText;

    const auto textHeight = static_cast< HPDF_REAL >( params.textParams.fontSize ) * 1.6f;

    if ( cursorY_ - textHeight < borderFieldBottom )
        newPage();

    float posX = borderFieldLeft;
    for ( int i = 0; i < cells.size(); ++i )
    {
        params.rect = Box2f( { posX, cursorY_ - textHeight }, { posX + columnsInfo_[i].width, cursorY_ } );
        std::string text = cells[i].toString( columnsInfo_[i].valueFormat );
        if ( tableCustomRule_ )
        {
            TextCellParams customParams = params;
            CellCustomParams cellParams = tableCustomRule_( rowCounter_, i, text );
            if ( cellParams.text.has_value() )
                text = *cellParams.text;
            if ( cellParams.colorText.has_value() )
                customParams.textParams.colorText = *cellParams.colorText;
            if ( cellParams.colorCellBg.has_value() )
                customParams.colorBackground = *cellParams.colorCellBg;
            if ( cellParams.colorCellBorder.has_value() )
                customParams.colorBorder = *cellParams.colorCellBorder;

            drawTextCell_( text, customParams );
        }
        else
            drawTextCell_( text, params );
        posX += columnsInfo_[i].width;
    }
    ++rowCounter_;
    cursorY_ -= textHeight;
    return {};
}

float Pdf::getTableTextWidth( const std::string& text )
{
    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetFontAndSize( state_->activePage, state_->defaultFont, params_.textSize ) );
    return MR_HPDF_CHECK_ERROR( HPDF_Page_TextWidth( state_->activePage, text.c_str() ) );
}

void Pdf::addText_( const std::string& text, const TextParams& textParams )
{
    if ( !checkDocument_( "add text" ) )
        return;

    if ( !textParams.colorText.a )
        return;

    int strNum = calcTextLinesCount_( text );
    const auto textHeight = static_cast< HPDF_REAL >( textParams.fontSize * strNum * lineSpacingScale );

    // TODO need add the ability to transfer text between pages
    if ( cursorY_ - textHeight < borderFieldBottom )
        newPage();

    Box2f rect;
    rect.include( { cursorX_, cursorY_ } );
    rect.include( { borderFieldRight, cursorY_ - textHeight } );
    drawTextRect_( text, rect, textParams );

    cursorY_ -= textHeight;
    moveCursorToNewLine();
}

void Pdf::drawTextRect_( const std::string& text, const Box2f& rect, const TextParams& params )
{
    if ( !checkDocument_( "draw text (rect)" ) )
        return;

    if ( text.empty() || !params.colorText.a )
        return;

    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetFontAndSize( state_->activePage, params.font, params.fontSize ) );
    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetTextLeading( state_->activePage, params.fontSize ) );

    const float verticalOffset = ( rect.size().y - params.fontSize ) / 2.f;

    Vector4f c = Vector4f( params.colorText );
    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetRGBFill( state_->activePage, c.x, c.y, c.z ) );

    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_BeginText( state_->activePage ) );
    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_TextRect( state_->activePage, rect.min.x, rect.max.y - verticalOffset,
        rect.max.x, rect.min.y + verticalOffset, text.c_str(), params.alignment, nullptr ) );
    MR_HPDF_CHECK_RES_STATUS( HPDF_Page_EndText( state_->activePage ) );


}

void Pdf::drawRect_( const Box2f& rect, const Color& fillColor, const Color& strokeColor )
{
    if ( !checkDocument_( "draw rect" ) )
        return;

    if ( fillColor.a )
    {
        Vector4f c = Vector4f( fillColor );
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetRGBFill( state_->activePage, c.x, c.y, c.z ) );
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_Rectangle( state_->activePage, rect.min.x, rect.min.y, rect.size().x, rect.size().y ) );
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_Fill( state_->activePage ) );
    }
    if ( strokeColor.a )
    {
        Vector4f c = Vector4f( strokeColor );
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_SetRGBStroke( state_->activePage, c.x, c.y, c.z ) );
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_Rectangle( state_->activePage, rect.min.x, rect.min.y, rect.size().x, rect.size().y ) );
        MR_HPDF_CHECK_RES_STATUS( HPDF_Page_Stroke( state_->activePage ) );
    }
}

void Pdf::drawTextCell_( const std::string& text, const TextCellParams& params )
{
    drawRect_( params.rect, params.colorBackground, params.colorBorder );
    drawTextRect_( text, params.rect, params.textParams );
}

void Pdf::reset_()
{
    HPDF_ResetError( state_->document );
    HPDF_Free( state_->document );
    state_->document = nullptr;
    state_->activePage = nullptr;
    state_->defaultFont = nullptr;
    state_->tableFont = nullptr;
}

int Pdf::calcTextLinesCount_( const std::string& text )
{
    HPDF_REAL r;
    HPDF_UINT substrStart = 0;
    int count = 0;
    for ( ; substrStart < text.size(); ++count )
    {
        HPDF_UINT lineSize = MR_HPDF_CHECK_ERROR( HPDF_Page_MeasureText( state_->activePage, text.data() + substrStart, pageWorkWidth, HPDF_TRUE, &r ) );
        if ( lineSize == 0 )
            break;
        substrStart += lineSize;
    }
    return count;
}

bool Pdf::checkDocument_( const std::string& logAction ) const
{
    if ( !state_->document )
    {
        spdlog::warn( "Pdf: Can't {}: no valid document", logAction );
        return false;
    }
    if ( !state_->activePage )
    {
        spdlog::warn( "Pdf: Can't {}: no valid page", logAction );
        return false;
    }

    return true;
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
