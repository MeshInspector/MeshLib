#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_PDF )
#include "MRPdf.h"
#include "MRSerializer.h"
#include "MRImageSave.h"
#include "MRVector2.h"
#include "MRImage.h"
#include "MRBox.h"
#include "MRStringConvert.h"
#include "MRGTest.h"
#include "MRPch/MRSpdlog.h"
#include "MRSystem.h"
#include <fstream>

#undef NOMINMAX

#pragma warning(push)
#pragma warning(disable:4464) //relative include path contains '..'
#pragma warning(disable:4800) //Implicit conversion from 'const int' to bool. Possible information loss
#pragma warning(disable:4986) //exception specification does not match previous declaration
#include <podofo/podofo.h>
#pragma warning(pop)

namespace MR
{

namespace
{
// size of A4 page in pixels (uses 72 PPI)
// https://www.papersizes.org/a-sizes-in-pixels.htm
// TODO need get this value from PoDoFo
constexpr double pageWidth = 595.;
constexpr double pageHeight = 842.;
constexpr double scaleFactor = 17. / 6.; // ~2.8(3)

constexpr double borderFieldLeft = 20 * scaleFactor;
constexpr double borderFieldRight = pageWidth - 10 * scaleFactor;
constexpr double borderFieldTop = pageHeight - 10 * scaleFactor;
constexpr double borderFieldBottom = 10 * scaleFactor;
constexpr double pageWorkWidth = borderFieldRight - borderFieldLeft;
constexpr double pageWorkHeight = borderFieldTop - borderFieldBottom;

constexpr double spacing = 6 * scaleFactor;

constexpr double textSpacing = 4 * scaleFactor;

constexpr double labelHeight = 10 * scaleFactor;
constexpr double marksHeight = 10 * scaleFactor;
constexpr double marksWidth = 30 * scaleFactor;

}

Pdf::Pdf( const std::filesystem::path& documentPath, const PdfParameters& params /*= PdfParameters()*/ ) :
filename_{ documentPath },
params_( params )
{
    cursorX_ = borderFieldLeft;
    cursorY_ = borderFieldTop;

    document_ = std::make_unique<PoDoFo::PdfMemDocument>();

    painter_ = std::make_unique<PoDoFo::PdfPainter>();
    if ( !painter_ )
    {
        spdlog::warn( "Can't create painter." );
        return;
    }
 #if PODOFO_VERSION >= 0x000a00
    activePage_ = &document_->GetPages().CreatePage( PoDoFo::PdfPage::CreateStandardPageSize( PoDoFo::PdfPageSize::A4 ) );
 #else
    activePage_ = document_->CreatePage( PoDoFo::PdfPage::CreateStandardPageSize( PoDoFo::ePdfPageSize_A4 ) );
#endif
    if ( !activePage_ )
    {
        spdlog::warn( "Can't create page." );
        return;
    }
#if PODOFO_VERSION >= 0x000a00
    painter_->SetCanvas( *activePage_ );
#else
    painter_->SetPage( activePage_ );
#endif

 #if PODOFO_VERSION >= 0x000a00
    auto fontpath = GetFontsDirectory() / "NotoSans-Regular.ttf";
    activeFont_ = &document_->GetFonts().GetOrCreateFont( fontpath.string() );
 #else
    activeFont_ = document_->CreateFont( params_.fontName.c_str() );
#endif
    if ( !activeFont_ )
    {
        spdlog::warn( "Can't found font : \"{}\"", params_.fontName );
        return;
    }
#if PODOFO_VERSION >= 0x000a00
    painter_->TextState.SetFont( *activeFont_, params_.textSize );
#else
    activeFont_->SetFontSize( params_.textSize );
    painter_->SetFont( activeFont_ );
#endif
}

Pdf::~Pdf()
{
    close();
}

void Pdf::addText( const std::string& text, bool isTitle /*= false*/ )
{
    if ( !checkDocument() || !activeFont_ )
        return;
#if PODOFO_VERSION >= 0x000a00
    painter_->TextState.SetFont( *activeFont_, isTitle ? params_.titleSize : params_.textSize  );
#else
    activeFont_->SetFontSize( isTitle ? params_.titleSize : params_.textSize );
    painter_->SetFont( activeFont_ );
#endif

    int strNum = 1;
    size_t pos = text.find( '\n', 0 );
    while ( pos != std::string::npos )
    {
        ++strNum;
        pos = text.find( '\n', pos + 1 );
    }

    const double textHeight = params_.textSize * strNum + textSpacing * 2.;

    if ( cursorY_ - textHeight < borderFieldBottom )
        newPage();
    
    cursorY_ -= textHeight;

#if PODOFO_VERSION >= 0x000a00 
    const PoDoFo::PdfHorizontalAlignment alignment = isTitle ? PoDoFo::PdfHorizontalAlignment::Center : PoDoFo::PdfHorizontalAlignment::Left;
#else
    const PoDoFo::EPdfAlignment alignment = isTitle ? PoDoFo::ePdfAlignment_Center : PoDoFo::ePdfAlignment_Left;
#endif

#if PODOFO_VERSION >= 0x000a00
    painter_->DrawTextMultiLine( text.c_str(), cursorX_, cursorY_, pageWorkWidth, textHeight,
        { .HorizontalAlignment = alignment, .VerticalAlignment = PoDoFo::PdfVerticalAlignment::Center } );
#else
    painter_->DrawMultiLineText( PoDoFo::PdfRect( cursorX_, cursorY_, pageWorkWidth, textHeight ), text.c_str(),
        alignment, PoDoFo::ePdfVerticalAlignment_Center );
#endif

    if ( cursorY_ - spacing < borderFieldBottom )
        newPage();
    else
        cursorY_ -= spacing;
}

void Pdf::addTextManual( const std::string& text, const Box2d& box, HorAlignment horAlignment, VertAlignment vertAlignment )
{
    if ( !checkDocument() || !activeFont_ )
        return;
#if PODOFO_VERSION >= 0x000a00
    painter_->TextState.SetFont( *activeFont_, params_.textSize  );
#else
    activeFont_->SetFontSize( params_.textSize );
    painter_->SetFont( activeFont_ );
#endif

#if PODOFO_VERSION >= 0x000a00
    PoDoFo::PdfHorizontalAlignment alignment = PoDoFo::PdfHorizontalAlignment::Center;
    switch ( horAlignment )
    {
    case MR::Pdf::HorAlignment::Left:
        alignment = PoDoFo::PdfHorizontalAlignment::Left;
        break;
    case MR::Pdf::HorAlignment::Right:
        alignment = PoDoFo::PdfHorizontalAlignment::Right;
        break;
    case MR::Pdf::HorAlignment::Center:
    default:
        break;
    }
#else
    PoDoFo::EPdfAlignment alignment = PoDoFo::ePdfAlignment_Center;
    switch ( horAlignment )
    {
    case MR::Pdf::HorAlignment::Left:
        alignment = PoDoFo::ePdfAlignment_Left;
        break;
    case MR::Pdf::HorAlignment::Right:
        alignment = PoDoFo::ePdfAlignment_Right;
        break;
    case MR::Pdf::HorAlignment::Center:
    default:
        break;
    }
#endif

#if PODOFO_VERSION >= 0x000a00
    PoDoFo::PdfVerticalAlignment verticalAlignment = PoDoFo::PdfVerticalAlignment::Center;
    switch ( vertAlignment )
    {
    case MR::Pdf::VertAlignment::Top:
        verticalAlignment = PoDoFo::PdfVerticalAlignment::Top;
        break;
    case MR::Pdf::VertAlignment::Bottom:
        verticalAlignment = PoDoFo::PdfVerticalAlignment::Bottom;
        break;
    case MR::Pdf::VertAlignment::Center:
    default:
        break;
    }
#else
    PoDoFo::EPdfVerticalAlignment verticalAlignment = PoDoFo::ePdfVerticalAlignment_Center;
    switch ( vertAlignment )
    {
    case MR::Pdf::VertAlignment::Top:
        verticalAlignment = PoDoFo::ePdfVerticalAlignment_Top;
        break;
    case MR::Pdf::VertAlignment::Bottom:
        verticalAlignment = PoDoFo::ePdfVerticalAlignment_Bottom;
        break;
    case MR::Pdf::VertAlignment::Center:
    default:
        break;
    }
#endif
#if PODOFO_VERSION >= 0x000a00
    painter_->DrawTextMultiLine( text.c_str(), box.min.x, box.min.y, box.size().x, box.size().y,
        { .HorizontalAlignment = alignment, .VerticalAlignment = verticalAlignment } );
#else
    painter_->DrawMultiLineText( PoDoFo::PdfRect( box.min.x, box.min.y, box.size().x, box.size().y ), text.c_str(),
        alignment, verticalAlignment );
#endif
}

void Pdf::addImageFromFile( const std::filesystem::path& imagePath, const std::string& caption /*= {}*/,
        const std::vector<std::pair<double, std::string>>& valuesMarks /*= {}*/ )
{
    if ( !checkDocument() || !activeFont_ )
        return;

    
#if PODOFO_VERSION >= 0x000a00
    std::unique_ptr<PoDoFo::PdfImage> pdfImage = document_->CreateImage();
    pdfImage->Load( imagePath.c_str() );
#else
    std::unique_ptr<PoDoFo::PdfImage> pdfImage = std::make_unique<PoDoFo::PdfImage>( document_.get() );
    pdfImage->LoadFromFile( imagePath.c_str() );
#endif

    const double additionalHeight = marksHeight * valuesMarks.empty() + labelHeight * caption.empty();
    const double scalingFactor = std::min( ( pageWorkHeight - additionalHeight ) / pdfImage->GetHeight(), pageWorkWidth / pdfImage->GetWidth() );
    const double scalingWidth = scalingFactor * pdfImage->GetWidth();
    const double scalingHeight = scalingFactor * pdfImage->GetHeight();

    if ( cursorY_ - scalingHeight - additionalHeight < borderFieldBottom )
        newPage();

    cursorY_ -= scalingHeight;

#if PODOFO_VERSION >= 0x000a00
    painter_->DrawImage( *pdfImage, cursorX_, cursorY_, scalingFactor, scalingFactor );
#else
    painter_->DrawImage( cursorX_, cursorY_, pdfImage.get(), scalingFactor, scalingFactor );
#endif

    if ( activeFont_ )
    {
        if ( !valuesMarks.empty() )
        {
            cursorY_ -= marksHeight;
            for ( auto& mark : valuesMarks )
            {
                const double posX = cursorX_ + scalingWidth * mark.first - marksWidth / 2.;
#if PODOFO_VERSION >= 0x000a00
                painter_->DrawTextMultiLine( mark.second.c_str(), posX , cursorY_, marksWidth, marksHeight,
                    { .HorizontalAlignment = PoDoFo::PdfHorizontalAlignment::Center , .VerticalAlignment = PoDoFo::PdfVerticalAlignment::Center } );
#else
                painter_->DrawMultiLineText( PoDoFo::PdfRect( posX , cursorY_, marksWidth, marksHeight ), mark.second.c_str(),
                    PoDoFo::ePdfAlignment_Center, PoDoFo::ePdfVerticalAlignment_Center );
#endif
            }
        }

        if ( !caption.empty() )
        {
            cursorY_ -= labelHeight;
#if PODOFO_VERSION >= 0x000a00
            painter_->DrawTextMultiLine( caption.c_str(), cursorX_, cursorY_, pageWorkWidth, labelHeight,
                { .HorizontalAlignment = PoDoFo::PdfHorizontalAlignment::Center , .VerticalAlignment = PoDoFo::PdfVerticalAlignment::Center } );
#else
            painter_->DrawMultiLineText( PoDoFo::PdfRect( cursorX_, cursorY_, pageWorkWidth, labelHeight ), caption.c_str(),
                PoDoFo::ePdfAlignment_Center, PoDoFo::ePdfVerticalAlignment_Center );
#endif
        }
    }

    if ( cursorY_ - spacing < borderFieldBottom )
        newPage();
    else
        cursorY_ -= spacing;
}

void Pdf::addImageFromFileManual( const std::filesystem::path& imagePath, const Box2d& box,
    HorAlignment horAlignment /*= HorAlignment::Center*/, VertAlignment vertAlignment /*= VertAlignment::Center*/ )
{
    if ( !checkDocument() )
        return;

#if PODOFO_VERSION >= 0x000a00
    std::unique_ptr<PoDoFo::PdfImage> pdfImage = document_->CreateImage();
    pdfImage->Load( imagePath.c_str() );
#else
    std::unique_ptr<PoDoFo::PdfImage> pdfImage = std::make_unique<PoDoFo::PdfImage>( document_.get() );
    pdfImage->LoadFromFile( imagePath.c_str() );
#endif

    const double scalingFactor = std::min( box.size().x / pdfImage->GetWidth(), box.size().y / pdfImage->GetHeight() );
    const double scalingWidth = scalingFactor * pdfImage->GetWidth();
    const double scalingHeight = scalingFactor * pdfImage->GetHeight();

    double posX = box.min.x + ( box.size().x - scalingWidth ) / 2.;
    switch ( horAlignment )
    {
    case MR::Pdf::HorAlignment::Left:
        posX = box.min.x;
        break;
    case MR::Pdf::HorAlignment::Right:
        posX = box.max.x - scalingWidth;
        break;
    case MR::Pdf::HorAlignment::Center:
    default:
        break;
    }

    double posY = box.min.y + ( box.size().y - scalingHeight ) / 2.;
    switch ( vertAlignment )
    {
    case MR::Pdf::VertAlignment::Top:
        posY = box.max.x - scalingHeight;
        break;
    case MR::Pdf::VertAlignment::Bottom:
        posY = box.min.x;
        break;
    case MR::Pdf::VertAlignment::Center:
    default:
        break;
    }
#if PODOFO_VERSION >= 0x000a00
    painter_->DrawImage( *pdfImage, posX, posY, scalingFactor, scalingFactor );
#else
    painter_->DrawImage( posX, posY, pdfImage.get(), scalingFactor, scalingFactor );
#endif
}

void Pdf::newPage()
{
    if ( !checkDocument() )
        return;

#if PODOFO_VERSION >= 0x000a00
    painter_->FinishDrawing();
    activePage_ = &document_->GetPages().CreatePage( PoDoFo::PdfPage::CreateStandardPageSize( PoDoFo::PdfPageSize::A4 ) );
#else
    painter_->FinishPage();
    activePage_ = document_->CreatePage( PoDoFo::PdfPage::CreateStandardPageSize( PoDoFo::ePdfPageSize_A4 ) );
#endif
    if ( !activePage_ )
    {
        spdlog::warn( "Can't create page." );
        return;
    }
#if PODOFO_VERSION >= 0x000a00
    painter_->SetCanvas( *activePage_ );
#else
    painter_->SetPage( activePage_ );
#endif

    cursorX_ = borderFieldLeft;
    cursorY_ = borderFieldTop;
}

void Pdf::close()
{
    if ( checkDocument() )
    {
        auto pathString = utf8string( filename_ );
#if PODOFO_VERSION >= 0x000a00
        painter_->FinishDrawing();
        document_->Save( pathString );
#else
        painter_->FinishPage();
        document_->Write( pathString.c_str() );
#endif
    }

    if ( document_ )
        document_.reset();
    if ( painter_ )
        painter_.reset();
    if ( activePage_ )
        activePage_ = nullptr;
    if ( activeFont_ )
        activeFont_ = nullptr;
}

TEST( MRMesh, Pdf )
{
    UniqueTemporaryFolder pathFolder( {} );
    Pdf pdfTest( pathFolder / std::filesystem::path( "test.pdf" ) );
    pdfTest.addText( "Test Title", true );
    pdfTest.addText( "Test text"
        "\nstring 1"
        "\nstring 2" );

    const int colorMapSizeX = int( pageWorkWidth);
    const int colorMapSizeY = int( 10 * scaleFactor );
    std::vector<Color> pixels( colorMapSizeX * colorMapSizeY );
    Color colorLeft = Color::blue();
    Color colorRight = Color::red();
    for ( int i = 0; i < colorMapSizeX; ++i )
    {
        for ( int j = 0; j < colorMapSizeY; ++j )
        {
            const float c = float( i ) / colorMapSizeX;
            pixels[i + j * colorMapSizeX] = ( 1 - c ) * colorLeft + c * colorRight;
        }
    }

    auto colorMapPath = pathFolder / std::filesystem::path( "color_map.png" );
    auto res = ImageSave::toAnySupportedFormat( { pixels, Vector2i( colorMapSizeX, colorMapSizeY ) }, colorMapPath );


    pdfTest.addImageFromFile( colorMapPath, "test image" );
    pdfTest.close();
}

}
#endif
