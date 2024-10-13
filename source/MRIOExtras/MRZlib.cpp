#include "MRZlib.h"
#ifndef MRIOEXTRAS_NO_ZLIB
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRFinally.h"

#include <zlib.h>

#include <cassert>

namespace
{

constexpr size_t cChunkSize = 256 * 1024; // 256 KiB

std::string zlibToString( int code )
{
    switch ( code )
    {
        case Z_OK:
            return "ok";
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

Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, int level )
{
    Buffer<char> inChunk( cChunkSize ), outChunk( cChunkSize );
    z_stream stream {
        .zalloc = Z_NULL,
        .zfree = Z_NULL,
        .opaque = Z_NULL,
    };
    int ret;
    if ( Z_OK != ( ret = deflateInit( &stream, level ) ) )
        return unexpected( zlibToString( ret ) );

    MR_FINALLY {
        deflateEnd( &stream );
    };

    while ( !in.eof() )
    {
        in.read( inChunk.data(), inChunk.size() );
        if ( in.bad() )
            return unexpected( "I/O error" );
        stream.next_in = reinterpret_cast<uint8_t*>( inChunk.data() );
        stream.avail_in = (unsigned)in.gcount();
        assert( stream.avail_in <= (unsigned)inChunk.size() );

        const auto flush = in.eof() ? Z_FINISH : Z_NO_FLUSH;
        do
        {
            stream.next_out = reinterpret_cast<uint8_t*>( outChunk.data() );
            stream.avail_out = (unsigned)outChunk.size();
            ret = deflate( &stream, flush );
            if ( Z_OK != ret && Z_STREAM_END != ret )
                return unexpected( zlibToString( ret ) );

            assert( stream.avail_out <= (unsigned)outChunk.size() );
            out.write( outChunk.data(), (unsigned)outChunk.size() - stream.avail_out );
            if ( out.bad() )
                return unexpected( "I/O error" );
        }
        while ( stream.avail_out == 0 );
    }

    return {};
}

Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out )
{
    Buffer<char> inChunk( cChunkSize ), outChunk( cChunkSize );
    z_stream stream {
        .zalloc = Z_NULL,
        .zfree = Z_NULL,
        .opaque = Z_NULL,
    };
    int ret;
    if ( Z_OK != ( ret = inflateInit( &stream ) ) )
        return unexpected( zlibToString( ret ) );

    MR_FINALLY {
        inflateEnd( &stream );
    };

    while ( !in.eof() )
    {
        in.read( inChunk.data(), inChunk.size() );
        if ( in.bad() )
            return unexpected( "I/O error" );
        stream.next_in = reinterpret_cast<uint8_t*>( inChunk.data() );
        stream.avail_in = (unsigned)in.gcount();
        assert( stream.avail_in <= (unsigned)inChunk.size() );

        do
        {
            stream.next_out = reinterpret_cast<uint8_t*>( outChunk.data() );
            stream.avail_out = (unsigned)outChunk.size();
            ret = inflate( &stream, Z_NO_FLUSH );
            if ( Z_OK != ret && Z_STREAM_END != ret )
                return unexpected( zlibToString( ret ) );

            assert( stream.avail_out <= (unsigned)outChunk.size() );
            out.write( outChunk.data(), (unsigned)outChunk.size() - stream.avail_out );
            if ( out.bad() )
                return unexpected( "I/O error" );

            if ( Z_STREAM_END == ret )
                return {};
        }
        while ( stream.avail_out == 0 );
    }

    return {};
}

} // namespace MR
#endif
