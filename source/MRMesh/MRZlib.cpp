#include "MRZlib.h"
#include "MRDeferred.h"

#include <zlib.h>

namespace
{

constexpr size_t cChunkSize = 256 * 1024; // 256 KiB

std::string zlibToString( int code )
{
    switch ( code )
    {
        case Z_OK:
            return "OK";
        case Z_STREAM_END:
            return "stream end";
        case Z_NEED_DICT:
            return "need dict";
        case Z_ERRNO:
            return "errno";
        case Z_STREAM_ERROR:
            return "stream error";
        case Z_DATA_ERROR:
            return "data error";
        case Z_MEM_ERROR:
            return "mem error";
        case Z_BUF_ERROR:
            return "buf error";
        case Z_VERSION_ERROR:
            return "version error";
        default:
            return "unknown code";
    }
}

} // namespace

namespace MR
{

VoidOrErrStr zlibCompressStream( std::istream& in, std::ostream& out, int level )
{
    char inChunk[cChunkSize], outChunk[cChunkSize];
    z_stream stream {
        .next_in = reinterpret_cast<uint8_t*>( inChunk ),
        .next_out = reinterpret_cast<uint8_t*>( outChunk ),
        .zalloc = Z_NULL,
        .zfree = Z_NULL,
        .opaque = Z_NULL,
    };
    int ret;
    if ( Z_OK != ( ret = deflateInit( &stream, level ) ) )
        return unexpected( zlibToString( ret ) );

    MR_DEFER_INLINE( deflateEnd( &stream ) )

    while ( !in.eof() )
    {
        in.read( inChunk, cChunkSize );
        if ( in.fail() )
            return unexpected( "I/O error" );
        stream.avail_in = in.gcount();

        const auto flush = in.eof() ? Z_FINISH : Z_NO_FLUSH;
        do
        {
            stream.avail_out = cChunkSize;
            ret = deflate( &stream, flush );
            if ( Z_OK != ret && Z_STREAM_END != ret )
                return unexpected( zlibToString( ret ) );

            out.write( outChunk, cChunkSize - stream.avail_out );
            if ( out.fail() )
                return unexpected( "I/O error" );
        }
        while ( stream.avail_out == 0 );
    }

    return {};
}

VoidOrErrStr zlibDecompressStream( std::istream& in, std::ostream& out )
{
    char inChunk[cChunkSize], outChunk[cChunkSize];
    z_stream stream {
        .next_in = reinterpret_cast<uint8_t*>( inChunk ),
        .next_out = reinterpret_cast<uint8_t*>( outChunk ),
        .zalloc = Z_NULL,
        .zfree = Z_NULL,
        .opaque = Z_NULL,
    };
    int ret;
    if ( Z_OK != ( ret = inflateInit( &stream ) ) )
        return unexpected( zlibToString( ret ) );

    MR_DEFER_INLINE( inflateEnd( &stream ) )

    while ( !in.eof() )
    {
        in.read( inChunk, cChunkSize );
        if ( in.fail() )
            return unexpected( "I/O error" );
        stream.avail_in = in.gcount();

        do
        {
            stream.avail_out = cChunkSize;
            ret = inflate( &stream, Z_NO_FLUSH );
            if ( Z_OK != ret && Z_STREAM_END != ret )
                return unexpected( zlibToString( ret ) );

            out.write( outChunk, cChunkSize - stream.avail_out );
            if ( out.fail() )
                return unexpected( "I/O error" );

            if ( Z_STREAM_END == ret )
                return {};
        }
        while ( stream.avail_out == 0 );
    }

    return {};
}

} // namespace MR
