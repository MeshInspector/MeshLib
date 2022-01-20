#include "MRImageSave.h"
#include "MRImage.h"
#include "MRFile.h"
#include "MRStringConvert.h"
#include <fstream>
#include <filesystem>
#include <tl/expected.hpp>
#include <string>
#include <libpng16/png.h>
#include <turbojpeg.h>

namespace MR
{
namespace ImageSave
{

const IOFilters Filters =
{
    {"Portable Network Graphics (.png)",  "*.png"},
    {"BitMap Picture (.bmp)",  "*.bmp"},
    {"JPEG (.jpg)",  "*.jpg"}
};

#pragma pack(push, 1)
struct BMPHeader
{
    const char sign[2] = {'B','M'};
    uint32_t size = 0;
    const uint32_t reserved = 0;
    const uint32_t dataOffset = 54;
    const uint32_t sizeOfHeader = 40;
    uint32_t width = 0;
    uint32_t height = 0;
    const uint16_t planes = 1;
    const uint16_t bitsPerPixel = 32;
    const uint32_t compression = 0;//no compression
    const uint32_t imageSize = 0; //can be 0 as far as there is no compression
    const uint32_t pixelsPerMeterX = 11811; // 300 dpi
    const uint32_t pixelsPerMeterY = 11811; // 300 dpi
    const uint32_t colorUsed = 0; // all
    const uint32_t importantColors = 0; // all
};
#pragma pack(pop)

tl::expected<void, std::string> toBmp( const Image& image, const std::filesystem::path& file )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    BMPHeader header;
    static_assert( sizeof( header ) == 54 );
    static_assert( sizeof( Color ) == 4 );
    header.size = uint32_t( image.pixels.size() ) * sizeof( Color ) + header.dataOffset;
    header.width = image.resolution.x;
    header.height = image.resolution.y;

    out.write( (const char*) &header, sizeof( header ) );
    for ( const auto& p : image.pixels )
    {
        out.write( (const char*) &p[2], 1 );
        out.write( (const char*) &p[1], 1 );
        out.write( (const char*) &p[0], 1 );
        out.write( (const char*) &p[3], 1 );
        if ( !out )
            return tl::make_unexpected( std::string( "Error saving image" ) );
    }

    return {};
}

tl::expected<void, std::string> toPng( const Image& image, const std::filesystem::path& file )
{
    std::ofstream fp( file, std::ios::binary );
    if ( !fp )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPng( image, fp );
}

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

tl::expected<void, std::string> toPng( const Image& image, std::ostream& os )
{
    WritePng png;
    if ( !png.pngPtr )
        return tl::make_unexpected( "Cannot create png" );

    if ( !png.infoPtr )
        return tl::make_unexpected( "Cannot create png info" );

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

struct JpegWriter
{
    JpegWriter()
    {
        tjInstance = tjInitCompress();
    }
    ~JpegWriter()
    {
        if ( tjInstance )
            tjDestroy( tjInstance );
        if ( jpegBuf )
            tjFree( jpegBuf );
    }
    unsigned char* jpegBuf{ nullptr };
    tjhandle tjInstance{ nullptr };
};

tl::expected<void, std::string> toJpeg( const Image& image, const std::filesystem::path& path )
{
    unsigned long jpegSize = 0;
    JpegWriter writer;

    if ( !writer.tjInstance )
        return tl::make_unexpected( "Cannot initialize JPEG compressor." );

    auto compressRes = tjCompress2( writer.tjInstance, ( unsigned char* )image.pixels.data(), image.resolution.x, 0, image.resolution.y, TJPF_RGBA, &writer.jpegBuf, &jpegSize, TJSAMP_444, 95, TJFLAG_BOTTOMUP );
    if ( compressRes != 0 )
        return tl::make_unexpected( "Error occurred while compressing image data." );

    std::ofstream outFile( path, std::ios::binary );
    if ( !outFile )
        return tl::make_unexpected( "Cannot write file " + path.string() );

    if ( !outFile.write( ( char* )writer.jpegBuf, jpegSize ) )
        return tl::make_unexpected( "Cannot write file " + path.string() );

    return {};
}

tl::expected<void, std::string> toAnySupportedFormat( const Image& image, const std::filesystem::path& file )
{
    auto ext = file.extension().u8string();
    for ( auto& c : ext )
        c = (char) tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == u8".png" )
        res = MR::ImageSave::toPng( image, file );
    else if ( ext == u8".bmp" )
        res = MR::ImageSave::toBmp( image, file );
    else if ( ext == u8".jpg" )
        res = MR::ImageSave::toJpeg( image, file );
    return res;
}

}
}
