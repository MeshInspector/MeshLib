#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <iostream>

namespace MR
{

/// parameters for zlibCompressStream
struct ZlibCompressParams
{
    /// compression level: 0 = no compression, 1 = the fastest but the most inefficient,
    /// 9 = the most efficient but the slowest; -1 = zlib's default
    int level = -1;
    /// if true, emit raw deflate (RFC 1951, no wrapper) — suitable for a ZIP entry;
    /// if false (default), emit zlib-wrapped output (RFC 1950)
    bool rawDeflate = false;
};

/// parameters for zlibDecompressStream
struct ZlibDecompressParams
{
    /// must match the framing of the compressed input:
    /// true for raw deflate (RFC 1951), false (default) for zlib-wrapped (RFC 1950)
    bool rawDeflate = false;
};

/**
 * @brief compress the input data using the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @param params - compression parameters (level, wire format)
 * @return nothing or error string
 */
MRMESH_API Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, const ZlibCompressParams& params );

/**
 * @brief compress the input data using the Deflate algorithm (zlib-wrapped, RFC 1950)
 * @note convenience overload; equivalent to `zlibCompressStream(in, out, { .level = level })`
 * @param level - compression level (see ZlibCompressParams::level)
 */
MRMESH_API Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, int level = -1 );

/**
 * @brief decompress the input data compressed with the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @param params - decompression parameters (wire format)
 * @return nothing or error string
 */
MRMESH_API Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out, const ZlibDecompressParams& params );

/**
 * @brief decompress the input data compressed with the Deflate algorithm (zlib-wrapped, RFC 1950)
 * @note convenience overload; equivalent to `zlibDecompressStream(in, out, {})`
 */
MRMESH_API Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out );

} // namespace MR
