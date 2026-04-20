#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <iostream>

namespace MR
{

/// wire format produced by zlibCompressStream / consumed by zlibDecompressStream
enum class DeflateFormat
{
    ZlibWrapper, ///< RFC 1950 — zlib header + Adler-32 trailer (default)
    Raw,         ///< RFC 1951 — raw deflate, no wrapper; suitable for ZIP entries
};

/**
 * @brief compress the input data using the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @param level - compression level (0 - no compression, 1 - the fastest but the most inefficient compression, 9 - the most efficient but the slowest compression)
 * @param format - wire format of the compressed output (see DeflateFormat)
 * @return nothing or error string
 */
MRMESH_API Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, int level = -1, DeflateFormat format = DeflateFormat::ZlibWrapper );

/**
 * @brief decompress the input data compressed with the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @param format - wire format of the compressed input; must match what produced it (see DeflateFormat)
 * @return nothing or error string
 */
MRMESH_API Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out, DeflateFormat format = DeflateFormat::ZlibWrapper );

} // namespace MR
