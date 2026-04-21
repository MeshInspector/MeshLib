#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <cstdint>
#include <iostream>

namespace MR
{

/// parameters shared by zlibCompressStream and zlibDecompressStream
struct ZlibParams
{
    /// wire format of the (de)compressed bytes:
    /// true → raw deflate (RFC 1951, no wrapper) — suitable for a ZIP entry;
    /// false (default) → zlib-wrapped (RFC 1950)
    bool rawDeflate = false;
};

/// statistics gathered during compression: CRC-32 of the uncompressed input and
/// the total numbers of bytes read from / written to the streams
struct ZlibCompressStats
{
    uint32_t crc32 = 0;            ///< CRC-32 of the uncompressed input
    size_t uncompressedSize = 0;   ///< total bytes read from the input stream
    size_t compressedSize = 0;     ///< total bytes written to the output stream
};

/// parameters for zlibCompressStream (adds a compression level on top of ZlibParams)
struct ZlibCompressParams : ZlibParams
{
    /// compression level: 0 = no compression, 1 = the fastest but the most inefficient,
    /// 9 = the most efficient but the slowest; -1 = zlib's default
    int level = -1;

    /// optional output; if non-null, the pointed-to object is populated with
    /// CRC-32 of the input and the uncompressed / compressed byte totals
    ZlibCompressStats* stats = nullptr;
};

/**
 * @brief compress the input data using the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @param params - compression parameters (wire format, level)
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
MRMESH_API Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out, const ZlibParams& params );

/**
 * @brief decompress the input data compressed with the Deflate algorithm (zlib-wrapped, RFC 1950)
 * @note convenience overload; equivalent to `zlibDecompressStream(in, out, {})`
 */
MRMESH_API Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out );

} // namespace MR
