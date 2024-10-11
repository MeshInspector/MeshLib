#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_ZLIB
#include "exports.h"

#include <MRMesh/MRExpected.h>

#include <iostream>

namespace MR
{

/**
 * @brief compress the input data using the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @param level - compression level (0 - no compression, 1 - the fastest but the most inefficient compression, 9 - the most efficient but the slowest compression)
 * @return nothing or error string
 */
MRIOEXTRAS_API Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, int level = -1 );

/**
 * /brief decompress the input data compressed using the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @return nothing or error string
 */
MRIOEXTRAS_API Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out );

} // namespace MR
#endif
