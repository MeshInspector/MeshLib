#include "MRImageLoad.h"
#include "MRImage.h"
#include "MRFile.h"
#include "MRStringConvert.h"
#include <fstream>
#include <filesystem>
#include <tl/expected.hpp>
#include <string>
#include <libpng16/png.h>

namespace MR
{
namespace ImageLoad
{

const IOFilters Filters =
{
    {"Portable Network Graphics (.png)",  "*.png"}
};

struct ReadPng
{
    ReadPng( const std::filesystem::path& file )
    {
        fp = fopen( file, "rb" );
        if ( !fp )
            return;
        pngPtr = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
        if ( !pngPtr )
            return;
        infoPtr = png_create_info_struct( pngPtr );
    }
    ~ReadPng()
    {
        if ( fp )
            fclose( fp );
        if ( pngPtr )
            png_destroy_read_struct( &pngPtr, &infoPtr, NULL );
    }

    png_structp pngPtr{ nullptr };
    png_infop infoPtr{ nullptr };
    FILE* fp{ nullptr };
};

tl::expected<Image, std::string> fromPng( const std::filesystem::path& file )
{
    ReadPng png( file );

    if ( !png.fp )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    if ( !png.pngPtr )
        return tl::make_unexpected( std::string( "Cannot read png " ) + utf8string( file ) );

    if ( !png.infoPtr )
        return tl::make_unexpected( std::string( "Cannot create png info" ) + utf8string( file ) );

    Image result;

    png_init_io( png.pngPtr, png.fp );

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
    if ( colorType != PNG_COLOR_TYPE_RGBA )
        return tl::make_unexpected( "Unsupported png color type" );

    result.resolution = Vector2i{ int( w ),int( h ) };
    result.pixels.resize( result.resolution.x * result.resolution.y );

    std::vector<unsigned char*> ptrs( result.resolution.y );
    for ( int i = 0; i < result.resolution.y; ++i )
        ptrs[result.resolution.y - i - 1] = ( unsigned char* )( result.pixels.data() + result.resolution.x * i );

    png_read_image( png.pngPtr, ptrs.data() );
    png_read_end( png.pngPtr, NULL );
    return result;
}

tl::expected<Image, std::string> fromAnySupportedFormat( const std::filesystem::path& file )
{
    auto ext = file.extension().u8string();
    for ( auto& c : ext )
        c = ( char )tolower( c );

    tl::expected<Image, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == u8".png" )
        return MR::ImageLoad::fromPng( file );
    return res;
}

}

}
