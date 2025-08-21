#if !defined __EMSCRIPTEN__ && !defined MRIOEXTRAS_NO_PDF
#include <MRIOExtras/MRPdf.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRUniqueTemporaryFolder.h>
#include <MRMesh/MRImage.h>
#include <MRMesh/MRImageSave.h>

namespace MR
{

TEST( MRMesh, Pdf )
{
    UniqueTemporaryFolder pathFolder( {} );
    Pdf pdfTest;
    pdfTest.addText( "Test Title", true );
    pdfTest.addText( "Test text"
        "\nstring 1"
        "\nstring 2" );

    constexpr float scaleFactor = 17.f / 6.f; // ~2.8(3)
    constexpr float pageWidth = 595.f;
    constexpr float borderFieldLeft = 20 * scaleFactor;
    constexpr float borderFieldRight = pageWidth - 10 * scaleFactor;
    constexpr float pageWorkWidth = borderFieldRight - borderFieldLeft;

    const int colorMapSizeX = int( pageWorkWidth );
    const int colorMapSizeY = int( 10 * scaleFactor );
    std::vector<Color> pixels( colorMapSizeX * colorMapSizeY );
    Color colorLeft = Color::blue();
    Color colorRight = Color::red();
    for ( int i = 0; i < colorMapSizeX; ++i )
    {
        for ( int j = 0; j < colorMapSizeY; ++j )
        {
            const float c = float( i ) / colorMapSizeX;
            pixels[i + j * colorMapSizeX] = lerp( colorLeft, colorRight, c );
        }
    }

    auto colorMapPath = pathFolder / std::filesystem::path( "color_map.png" );
    auto res = ImageSave::toAnySupportedFormat( { pixels, Vector2i( colorMapSizeX, colorMapSizeY ) }, colorMapPath );

    pdfTest.addImageFromFile( colorMapPath, { {-1, 0}, "test image", true } );
    pdfTest.saveToFile( pathFolder / std::filesystem::path( "test.pdf" ) );
}

} //namespace MR

#endif // !defined __EMSCRIPTEN__ && !defined MRIOEXTRAS_NO_PDF
