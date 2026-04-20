#include "MRZlib.h"
#include "MRBuffer.h"
#include "MRFinally.h"

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

Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, int level, bool rawDeflate )
{
    Buffer<char> inChunk( cChunkSize ), outChunk( cChunkSize );
    z_stream stream {
        .zalloc = Z_NULL,
        .zfree = Z_NULL,
        .opaque = Z_NULL,
    };
    int ret;
    // windowBits = 15 → zlib wrapper (RFC 1950); -15 → raw deflate (RFC 1951, no wrapper).
    // memLevel = 8 → zlib's default internal-state size.
    const int windowBits = rawDeflate ? -15 : 15;
    if ( Z_OK != ( ret = deflateInit2( &stream, level, Z_DEFLATED, windowBits, 8, Z_DEFAULT_STRATEGY ) ) )
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

Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out, bool rawDeflate )
{
    Buffer<char> inChunk( cChunkSize ), outChunk( cChunkSize );
    z_stream stream {
        .zalloc = Z_NULL,
        .zfree = Z_NULL,
        .opaque = Z_NULL,
    };
    int ret;
    // windowBits = 15 → expect zlib wrapper (RFC 1950); -15 → expect raw deflate (RFC 1951).
    const int windowBits = rawDeflate ? -15 : 15;
    if ( Z_OK != ( ret = inflateInit2( &stream, windowBits ) ) )
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
