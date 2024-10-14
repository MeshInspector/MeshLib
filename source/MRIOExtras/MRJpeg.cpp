#include "MRJpeg.h"
#ifndef MRIOEXTRAS_NO_JPEG
#include <MRMesh/MRBuffer.h>
#include <MRMesh/MRIOFormatsRegistry.h>
#include <MRMesh/MRIOParsing.h>
#include <MRMesh/MRStringConvert.h>

#include <turbojpeg.h>

#include <fstream>

namespace
{

struct JpegReader
{
    JpegReader()
    {
        tjInstance = tjInitDecompress();
    }
    ~JpegReader()
    {
        if ( tjInstance )
            tjDestroy( tjInstance );
    }
    tjhandle tjInstance{ nullptr };
};

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

} // namespace

namespace MR
{

namespace ImageLoad
{

Expected<Image> fromJpeg( const std::filesystem::path& path )
{
    std::ifstream in( path, std::ios::binary );
    if ( !in )
        return unexpected( "Cannot open file " + utf8string( path ) );

    return addFileNameInError( fromJpeg( in ), path );
}

Expected<Image> fromJpeg( std::istream& in )
{
    return
        readCharBuffer( in )
        .and_then( [] ( auto&& buffer )
        {
            return fromJpeg( buffer.data(), buffer.size() );
        } );
}

Expected<Image> fromJpeg( const char* data, size_t size )
{
    JpegReader reader;
    if ( !reader.tjInstance )
        return unexpected( "Cannot initialize JPEG decompressor" );

    int width, height, jpegSubsamp, jpegColorspace;
    auto res = tjDecompressHeader3( reader.tjInstance, ( const unsigned char* )data, ( unsigned long )size, &width, &height, &jpegSubsamp, &jpegColorspace );
    if ( res != 0 )
        return unexpected( "Failed to decompress JPEG header" );

    Image image;
    image.pixels.resize( width * height );
    image.resolution = { width, height };
    res = tjDecompress2( reader.tjInstance, ( const unsigned char* )data, ( unsigned long )size, reinterpret_cast< unsigned char* >( image.pixels.data() ), width, 0, height, TJPF_RGBA, TJFLAG_BOTTOMUP );
    if ( res != 0 )
        return unexpected( "Failed to decompress JPEG file" );

    return image;
}

MR_ADD_IMAGE_LOADER_WITH_PRIORITY( IOFilter( "JPEG (.jpg,.jpeg)", "*.jpg;*.jpeg" ), fromJpeg, -1 )

} // namespace ImageLoad

namespace ImageSave
{

Expected<void> toJpeg( const Image& image, const std::filesystem::path& path )
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

MR_ADD_IMAGE_SAVER_WITH_PRIORITY( IOFilter( "JPEG (.jpg)", "*.jpg" ), toJpeg, -1 )

} // namespace ImageSave

} // namespace MR
#endif
