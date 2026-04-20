#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <iostream>

namespace MR
{

/// Wire format produced by zlibCompressStream / consumed by zlibDecompressStream.
/// Enumerator values are the zlib `windowBits` argument for the chosen format
/// (positive = zlib wrapper per RFC 1950, negative = raw deflate per RFC 1951; magnitude = log2 window size).
enum class DeflateFormat : int
{
    Zlib = 15,  ///< RFC 1950 — zlib header + Adler-32 trailer (default)
    Raw  = -15, ///< RFC 1951 — raw deflate, no wrapper; suitable for ZIP entries
};

/**
 * @brief compress the input data using the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @param level - compression level (0 - no compression, 1 - the fastest but the most inefficient compression, 9 - the most efficient but the slowest compression)
 * @param format - wire format of the compressed output (see DeflateFormat)
 * @return nothing or error string
 */
MRMESH_API Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, int level = -1, DeflateFormat format = DeflateFormat::Zlib );

/**
 * @brief decompress the input data compressed with the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @param format - wire format of the compressed input; must match what produced it (see DeflateFormat)
 * @return nothing or error string
 */
MRMESH_API Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out, DeflateFormat format = DeflateFormat::Zlib );

} // namespace MR
