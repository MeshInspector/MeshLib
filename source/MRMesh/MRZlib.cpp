#include "MRZlib.h"
#include "MRBuffer.h"
#include "MRFinally.h"

// zlib-ng in native mode: the zng_ prefix on every symbol keeps it ABI-
// distinct from stock zlib (which libzip still links as before), and
// <zlib-ng.h> re-exports the MAX_WBITS / Z_* constants we need under the
// same spelling as <zlib.h>, so this TU doesn't include stock zlib's
// header at all.
#include <zlib-ng.h>

#include <cassert>
#include <cstdint>

namespace
{

constexpr size_t cChunkSize = 256 * 1024; // 256 KiB

// windowBits is sign-encoded the same way zlib documents it: positive =
// zlib wrapper (RFC 1950), negative = raw deflate (RFC 1951, no wrapper).
// Magnitude is log2(window size); MAX_WBITS = 15 gives a 32 KiB window.
// zlib-ng re-exports the same MAX_WBITS macro through zconf-ng.h, so we
// can use it here without pulling in stock zlib's headers.
constexpr int kZlibWrapperBits = MAX_WBITS;
constexpr int kRawDeflateBits  = -MAX_WBITS;

// memLevel controls the compressor's internal state size. 8 is zlib's and
// zlib-ng's shared default (matches the old MRZlib choice).
constexpr int kDefaultMemLevel = 8;

std::string zngToString( int code )
{
    switch ( code )
    {
        case Z_OK:             return "ok";
        case Z_STREAM_END:     return "stream end";
        case Z_NEED_DICT:      return "need dict";
        case Z_ERRNO:          return "errno";
        case Z_STREAM_ERROR:   return "stream error";
        case Z_DATA_ERROR:     return "data error";
        case Z_MEM_ERROR:      return "mem error";
        case Z_BUF_ERROR:      return "buf error";
        case Z_VERSION_ERROR:  return "version error";
        default:               return "unknown code";
    }
}

int windowBitsFor( bool rawDeflate )
{
    return rawDeflate ? kRawDeflateBits : kZlibWrapperBits;
}

} // namespace

namespace MR
{

Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, const ZlibCompressParams& params )
{
    Buffer<char> inChunk( cChunkSize ), outChunk( cChunkSize );
    zng_stream stream{};
    int ret;
    if ( Z_OK != ( ret = zng_deflateInit2( &stream, params.level, Z_DEFLATED,
                                           windowBitsFor( params.rawDeflate ),
                                           kDefaultMemLevel, Z_DEFAULT_STRATEGY ) ) )
        return unexpected( zngToString( ret ) );

    MR_FINALLY {
        zng_deflateEnd( &stream );
    };

    if ( params.stats )
        *params.stats = {};

    while ( !in.eof() )
    {
        in.read( inChunk.data(), static_cast<std::streamsize>( inChunk.size() ) );
        if ( in.bad() )
            return unexpected( "I/O error" );
        stream.next_in = reinterpret_cast<uint8_t*>( inChunk.data() );
        stream.avail_in = static_cast<unsigned>( in.gcount() );
        assert( stream.avail_in <= static_cast<unsigned>( inChunk.size() ) );

        if ( params.stats )
        {
            params.stats->crc32 = static_cast<uint32_t>(
                zng_crc32( params.stats->crc32, stream.next_in, stream.avail_in ) );
            params.stats->uncompressedSize += stream.avail_in;
        }

        const int flush = in.eof() ? Z_FINISH : Z_NO_FLUSH;
        do
        {
            stream.next_out = reinterpret_cast<uint8_t*>( outChunk.data() );
            stream.avail_out = static_cast<unsigned>( outChunk.size() );
            ret = zng_deflate( &stream, flush );
            if ( Z_OK != ret && Z_STREAM_END != ret )
                return unexpected( zngToString( ret ) );

            assert( stream.avail_out <= static_cast<unsigned>( outChunk.size() ) );
            const unsigned written = static_cast<unsigned>( outChunk.size() ) - stream.avail_out;
            out.write( outChunk.data(), written );
            if ( out.bad() )
                return unexpected( "I/O error" );
            if ( params.stats )
                params.stats->compressedSize += written;
        }
        while ( stream.avail_out == 0 );
    }

    return {};
}

Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, int level )
{
    return zlibCompressStream( in, out, ZlibCompressParams{ .level = level } );
}

Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out, const ZlibParams& params )
{
    Buffer<char> inChunk( cChunkSize ), outChunk( cChunkSize );
    zng_stream stream{};
    int ret;
    if ( Z_OK != ( ret = zng_inflateInit2( &stream, windowBitsFor( params.rawDeflate ) ) ) )
        return unexpected( zngToString( ret ) );

    MR_FINALLY {
        zng_inflateEnd( &stream );
    };

    while ( !in.eof() )
    {
        in.read( inChunk.data(), static_cast<std::streamsize>( inChunk.size() ) );
        if ( in.bad() )
            return unexpected( "I/O error" );
        stream.next_in = reinterpret_cast<uint8_t*>( inChunk.data() );
        stream.avail_in = static_cast<unsigned>( in.gcount() );
        assert( stream.avail_in <= static_cast<unsigned>( inChunk.size() ) );

        do
        {
            stream.next_out = reinterpret_cast<uint8_t*>( outChunk.data() );
            stream.avail_out = static_cast<unsigned>( outChunk.size() );
            ret = zng_inflate( &stream, Z_NO_FLUSH );
            if ( Z_OK != ret && Z_STREAM_END != ret )
                return unexpected( zngToString( ret ) );

            assert( stream.avail_out <= static_cast<unsigned>( outChunk.size() ) );
            out.write( outChunk.data(), static_cast<unsigned>( outChunk.size() ) - stream.avail_out );
            if ( out.bad() )
                return unexpected( "I/O error" );

            if ( Z_STREAM_END == ret )
                return {};
        }
        while ( stream.avail_out == 0 );
    }

    return {};
}

Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out )
{
    return zlibDecompressStream( in, out, ZlibParams{} );
}

} // namespace MR
