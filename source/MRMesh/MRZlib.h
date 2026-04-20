#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <iostream>

namespace MR
{

/**
 * @brief compress the input data using the raw Deflate algorithm (RFC 1951, no zlib/gzip wrapper)
 * @param in - input data stream
 * @param out - output data stream
 * @param level - compression level (0 - no compression, 1 - the fastest but the most inefficient compression, 9 - the most efficient but the slowest compression)
 * @return nothing or error string
 */
MRMESH_API Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, int level = -1 );

/**
 * @brief decompress the input data compressed with the raw Deflate algorithm (RFC 1951, no zlib/gzip wrapper)
 * @param in - input data stream
 * @param out - output data stream
 * @return nothing or error string
 */
MRMESH_API Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out );

} // namespace MR
