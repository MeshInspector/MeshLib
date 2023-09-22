#include "MRImageSave.h"
#include "MRImage.h"
#include "MRFile.h"
#include "MRStringConvert.h"
#include <fstream>
#include <filesystem>

#ifndef __EMSCRIPTEN__
#ifndef MRMESH_NO_PNG
#include <libpng16/png.h>
#endif
#ifndef MRMESH_NO_JPEG
#include <turbojpeg.h>
#endif
#ifndef MRMESH_NO_TIFF
#include "MRTiffIO.h"
#endif
#endif

namespace MR
{
namespace ImageSave
{

const IOFilters Filters =
{
#ifndef __EMSCRIPTEN__
#ifndef MRMESH_NO_PNG
    {"Portable Network Graphics (.png)",  "*.png"},
#endif
#ifndef MRMESH_NO_JPEG
    {"JPEG (.jpg)",  "*.jpg"},
#endif
#ifndef MRMESH_NO_TIFF
    {"TIFF (.tif)",  "*.tif"},
#endif
#ifndef MRMESH_NO_TIFF
    {"TIFF (.tiff)",  "*.tiff"},
#endif
#endif
    {"BitMap Picture (.bmp)",  "*.bmp"},
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

VoidOrErrStr toBmp( const Image& image, const std::filesystem::path& file )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

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
            return unexpected( std::string( "Error saving image" ) );
    }

    return {};
}

#ifndef __EMSCRIPTEN__

#ifndef MRMESH_NO_PNG
VoidOrErrStr toPng( const Image& image, const std::filesystem::path& file )
{
    std::ofstream fp( file, std::ios::binary );
    if ( !fp )
        return unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

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

VoidOrErrStr toPng( const Image& image, std::ostream& os )
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
#endif

#ifndef MRMESH_NO_JPEG
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

VoidOrErrStr toJpeg( const Image& image, const std::filesystem::path& path )
{
    unsigned long jpegSize = 0;
    JpegWriter writer;

    if ( !writer.tjInstance )
        return unexpected( "Cannot initialize JPEG compressor." );

    auto compressRes = tjCompress2( writer.tjInstance, ( unsigned char* )image.pixels.data(), image.resolution.x, 0, image.resolution.y, TJPF_RGBA, &writer.jpegBuf, &jpegSize, TJSAMP_444, 95, TJFLAG_BOTTOMUP );
    if ( compressRes != 0 )
        return unexpected( "Error occurred while compressing image data." );

    std::ofstream outFile( path, std::ios::binary );
    if ( !outFile )
        return unexpected( "Cannot write file " + utf8string( path ) );

    if ( !outFile.write( ( char* )writer.jpegBuf, jpegSize ) )
        return unexpected( "Cannot write file " + utf8string( path ) );

    return {};
}
#endif

#ifndef MRMESH_NO_TIFF
VoidOrErrStr toTiff( const Image& image, const std::filesystem::path& path )
{
    BaseTiffParameters btParams;
    btParams.bytesPerSample = 1;
    btParams.sampleType = BaseTiffParameters::SampleType::Uint;
    btParams.valueType = BaseTiffParameters::ValueType::RGBA;
    btParams.imageSize = image.resolution;
    return writeRawTiff( ( const uint8_t* )image.pixels.data(), path, btParams );
}
#endif
#endif



VoidOrErrStr toAnySupportedFormat( const Image& image, const std::filesystem::path& file )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    VoidOrErrStr res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".bmp" )
        res = MR::ImageSave::toBmp( image, file );
#ifndef __EMSCRIPTEN__
#ifndef MRMESH_NO_PNG
    else if ( ext == ".png" )
        res = MR::ImageSave::toPng( image, file );
#endif
#ifndef MRMESH_NO_JPEG
    else if ( ext == ".jpg" )
        res = MR::ImageSave::toJpeg( image, file );
#endif
#ifndef MRMESH_NO_TIFF
    else if ( ext == ".tif" || ext == ".tiff" )
        res = MR::ImageSave::toTiff( image, file );
#endif
#endif
    return res;
}

}
}
