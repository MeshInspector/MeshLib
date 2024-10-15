#include "MRPng.h"
#ifndef MRIOEXTRAS_NO_PNG

#include <MRMesh/MRIOFormatsRegistry.h>
#include <MRMesh/MRStringConvert.h>

#ifdef __EMSCRIPTEN__
#include <png.h>
#else
#include <libpng16/png.h>
#endif

#include <fstream>

namespace
{

struct ReadPng
{
    ReadPng( std::istream& in )
    {
        pngPtr = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
        if ( !pngPtr )
            return;
        infoPtr = png_create_info_struct( pngPtr );

        png_set_read_fn( pngPtr, ( png_voidp )&in, userReadData );
    }

    ~ReadPng()
    {
        if ( pngPtr )
            png_destroy_read_struct( &pngPtr, &infoPtr, NULL );
    }

    static void userReadData( png_structp pngPtr, png_bytep data, png_size_t length )
    {
        png_voidp a = png_get_io_ptr( pngPtr );
        ( ( std::istream* )a )->read( ( char* )data, length );
    }

    png_structp pngPtr{ nullptr };
    png_infop infoPtr{ nullptr };
    png_colorp palette{ nullptr };
    int paletteSize{ 0 };
    png_bytep alphaPalette{ nullptr };
    int alphaPaletteSize{ 0 };
};

struct WritePng
{
    WritePng()
    {
        pngPtr = png_create_write_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
        if ( !pngPtr )
            return;
        infoPtr = png_create_info_struct( pngPtr );
    }
    ~WritePng()
    {
        if ( pngPtr )
            png_destroy_write_struct( &pngPtr, &infoPtr );
    }

    png_structp pngPtr{ nullptr };
    png_infop infoPtr{ nullptr };
};

static void write_to_png( png_structp png_ptr, png_bytep data, png_size_t length )
{
    std::ostream* stream = ( std::ostream* )png_get_io_ptr( png_ptr );
    stream->write( ( char* )data, length );
}

static void flush_png( png_structp png_ptr )
{
    std::ostream* stream = ( std::ostream* )png_get_io_ptr( png_ptr );
    stream->flush();
}

} // namespace

namespace MR
{

namespace ImageLoad
{

Expected<Image> fromPng( const std::filesystem::path& path )
{
    std::ifstream in( path, std::ios::binary );
    if ( !in )
        return unexpected( "Cannot open file " + utf8string( path ) );

    return addFileNameInError( fromPng( in ), path );
}

Expected<MR::Image> fromPng( std::istream& in )
{
    ReadPng png( in );

    if ( !png.pngPtr )
        return unexpected( std::string( "Cannot read png" ) );

    if ( !png.infoPtr )
        return unexpected( std::string( "Cannot create png info" ) );

    Image result;

    char sign[8];
    if ( !in.read( sign, 8 ) )
        return unexpected( "Cannot read png signature" );
    in.seekg( 0 );

    if ( png_sig_cmp( ( png_const_bytep )sign, 0, 8 ) != 0 )
        return unexpected( "Invalid png signature" );

    unsigned w{ 0 }, h{ 0 };
    int depth{ 0 }; // 8
    int colorType{ 0 }; // PNG_COLOR_TYPE_RGBA
    int interlace{ 0 }; // PNG_INTERLACE_NONE
    int compression{ 0 }; // PNG_COMPRESSION_TYPE_DEFAULT
    int filter{ 0 }; // PNG_FILTER_TYPE_DEFAULT
    // Read info
    png_read_info( png.pngPtr, png.infoPtr );
    png_get_IHDR(
        png.pngPtr,
        png.infoPtr,
        &w, &h,
        &depth,
        &colorType,
        &interlace,
        &compression,
        &filter
    );

    if ( depth != 8 )
        return unexpected( std::string( "Unsupported png depth: " ) + std::to_string( depth ) );

    result.resolution = Vector2i{ int( w ),int( h ) };
    result.pixels.resize( result.resolution.x * result.resolution.y );

    std::vector<unsigned char*> ptrs( result.resolution.y );
    if ( colorType == PNG_COLOR_TYPE_RGBA )
    {
        for ( int i = 0; i < result.resolution.y; ++i )
            ptrs[result.resolution.y - i - 1] = ( unsigned char* )( result.pixels.data() + result.resolution.x * i );
        png_read_image( png.pngPtr, ptrs.data() );
    }
    else if ( colorType == PNG_COLOR_TYPE_RGB )
    {
        std::vector<Vector3<unsigned char>> rawPixels( result.resolution.x * result.resolution.y );
        for ( int i = 0; i < result.resolution.y; ++i )
            ptrs[result.resolution.y - i - 1] = ( unsigned char* )( rawPixels.data() + result.resolution.x * i );
        png_read_image( png.pngPtr, ptrs.data() );
        for ( size_t i = 0; i < result.pixels.size(); ++i )
            result.pixels[i] = Color( rawPixels[i] );
    }
    else if ( colorType == PNG_COLOR_TYPE_PALETTE )
    {
        png_get_PLTE( png.pngPtr, png.infoPtr, &png.palette, &png.paletteSize );
        png_get_tRNS( png.pngPtr, png.infoPtr, &png.alphaPalette, &png.alphaPaletteSize, nullptr );

        std::vector<Color> palette( png.paletteSize );
        for ( int i = 0; i < png.paletteSize; ++i )
        {
            palette[i] = Color( png.palette[i].red, png.palette[i].green, png.palette[i].blue );
            if ( png.alphaPalette && i < png.alphaPaletteSize )
                palette[i].a = png.alphaPalette[i];
        }

        std::vector<unsigned char> rawPixels( result.resolution.x * result.resolution.y );
        for ( int i = 0; i < result.resolution.y; ++i )
            ptrs[result.resolution.y - i - 1] = ( unsigned char* )( rawPixels.data() + result.resolution.x * i );

        png_read_image( png.pngPtr, ptrs.data() );
        for ( int i = 0; i < result.resolution.y; ++i )
            for ( int j = 0; j < result.resolution.x; ++j )
            {
                result.pixels[i * result.resolution.x + j] = palette[rawPixels[i * result.resolution.x + j]];
            }
    }
    else
        return unexpected( "Unsupported png color type" );

    png_read_end( png.pngPtr, NULL );
    return result;
}

MR_ADD_IMAGE_LOADER_WITH_PRIORITY( IOFilter( "Portable Network Graphics (.png)", "*.png" ), fromPng, -2 )

} // namespace ImageLoad

namespace ImageSave
{

Expected<void> toPng( const Image& image, const std::filesystem::path& file )
{
    std::ofstream fp( file, std::ios::binary );
    if ( !fp )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPng( image, fp );
}

Expected<void> toPng( const Image& image, std::ostream& os )
{
    WritePng png;
    if ( !png.pngPtr )
        return unexpected( "Cannot create png" );

    if ( !png.infoPtr )
        return unexpected( "Cannot create png info" );

    png_set_write_fn( png.pngPtr, &os, write_to_png, flush_png );

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
        png.pngPtr,
        png.infoPtr,
        image.resolution.x, image.resolution.y,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info( png.pngPtr, png.infoPtr );

    std::vector<unsigned char*> ptrs( image.resolution.y );
    for ( int i = 0; i < image.resolution.y; ++i )
        ptrs[image.resolution.y - i - 1] = ( unsigned char* )( image.pixels.data() + image.resolution.x * i );

    png_write_image( png.pngPtr, ptrs.data() );
    png_write_end( png.pngPtr, NULL );
    return {};
}

MR_ADD_IMAGE_SAVER_WITH_PRIORITY( IOFilter( "Portable Network Graphics (.png)", "*.png" ), toPng, -2 )

} // namespace ImageSave

} // namespace MR
#endif