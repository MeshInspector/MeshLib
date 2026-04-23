#include "MRZlib.h"
#include "MRBuffer.h"
#include "MRFinally.h"

#include <libdeflate.h>
#include <zlib.h>

#include <cassert>
#include <cstdint>
#include <vector>

namespace
{

constexpr size_t cChunkSize = 256 * 1024; // 256 KiB

// zlib's `windowBits` argument is sign-encoded: positive = zlib wrapper (RFC 1950);
// negative = raw deflate (RFC 1951, no wrapper). Magnitude is log2(window size);
// MAX_WBITS = 15 gives a 32 KiB window.
constexpr int kZlibWrapperBits = MAX_WBITS;
constexpr int kRawDeflateBits  = -MAX_WBITS;

// memLevel controls zlib's internal state size. 8 is zlib's default (its internal
// DEF_MEM_LEVEL in deflate.c is not exported; redeclared here).
constexpr int kDefaultMemLevel = 8;
static_assert( kDefaultMemLevel <= MAX_MEM_LEVEL );

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

int windowBitsFor( bool rawDeflate )
{
    return rawDeflate ? kRawDeflateBits : kZlibWrapperBits;
}

} // namespace

namespace MR
{

Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, const ZlibCompressParams& params )
{
    // libdeflate exposes only a whole-buffer compression API, so drain the
    // input stream into memory first. Memory ceiling = input size; callers
    // that need streaming compression of very large inputs should chunk at
    // a layer above this function (only this overload takes the fast path;
    // the other direction still streams in zlibDecompressStream below).
    std::vector<uint8_t> inBuf;
    while ( !in.eof() )
    {
        const size_t offset = inBuf.size();
        inBuf.resize( offset + cChunkSize );
        in.read( reinterpret_cast<char*>( inBuf.data() + offset ), cChunkSize );
        if ( in.bad() )
            return unexpected( "I/O error" );
        inBuf.resize( offset + static_cast<size_t>( in.gcount() ) );
    }

    // libdeflate accepts levels 1-12. Map the zlib-style conventions we
    // inherit from ZlibCompressParams: -1 (default) -> libdeflate 6, which
    // matches zlib's default; 0 (zlib's "store-only") -> libdeflate 1
    // because libdeflate has no stored-only mode. The ±4-byte tolerance on
    // ZlibCompressStats's size check absorbs the resulting minor differences
    // versus the stock-zlib reference blobs.
    int level = params.level;
    if ( level < 0 )
        level = 6;
    else if ( level == 0 )
        level = 1;
    else if ( level > 12 )
        level = 12;

    if ( params.stats )
    {
        params.stats->crc32 = libdeflate_crc32( 0, inBuf.data(), inBuf.size() );
        params.stats->uncompressedSize = inBuf.size();
        params.stats->compressedSize = 0;
    }

    libdeflate_compressor* comp = libdeflate_alloc_compressor( level );
    if ( !comp )
        return unexpected( "libdeflate_alloc_compressor failed" );
    MR_FINALLY {
        libdeflate_free_compressor( comp );
    };

    const bool raw = params.rawDeflate;
    const size_t bound = raw
        ? libdeflate_deflate_compress_bound( comp, inBuf.size() )
        : libdeflate_zlib_compress_bound( comp, inBuf.size() );
    std::vector<uint8_t> outBuf( bound );

    const size_t produced = raw
        ? libdeflate_deflate_compress( comp, inBuf.data(), inBuf.size(), outBuf.data(), outBuf.size() )
        : libdeflate_zlib_compress( comp, inBuf.data(), inBuf.size(), outBuf.data(), outBuf.size() );
    if ( produced == 0 )
        return unexpected( "libdeflate compression failed" );

    out.write( reinterpret_cast<const char*>( outBuf.data() ), static_cast<std::streamsize>( produced ) );
    if ( out.bad() )
        return unexpected( "I/O error" );

    if ( params.stats )
        params.stats->compressedSize = produced;

    return {};
}

Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, int level )
{
    return zlibCompressStream( in, out, ZlibCompressParams{ .level = level } );
}

Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out, const ZlibParams& params )
{
    Buffer<char> inChunk( cChunkSize ), outChunk( cChunkSize );
    z_stream stream {
        .zalloc = Z_NULL,
        .zfree = Z_NULL,
        .opaque = Z_NULL,
    };
    int ret;
    if ( Z_OK != ( ret = inflateInit2( &stream, windowBitsFor( params.rawDeflate ) ) ) )
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

Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out )
{
    return zlibDecompressStream( in, out, ZlibParams{} );
}

} // namespace MR
