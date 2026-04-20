#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"

#include <iostream>

namespace MR
{

/**
 * @brief compress the input data using the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @param level - compression level (0 - no compression, 1 - the fastest but the most inefficient compression, 9 - the most efficient but the slowest compression)
 * @param rawDeflate - if true, output is raw deflate (RFC 1951, no wrapper) suitable for a ZIP entry;
 *                    if false (default), output is zlib-wrapped (RFC 1950) — same as the original behaviour
 * @return nothing or error string
 */
MRMESH_API Expected<void> zlibCompressStream( std::istream& in, std::ostream& out, int level = -1, bool rawDeflate = false );

/**
 * @brief decompress the input data compressed with the Deflate algorithm
 * @param in - input data stream
 * @param out - output data stream
 * @param rawDeflate - must match the framing of the input: true for raw deflate (RFC 1951),
 *                    false (default) for zlib-wrapped (RFC 1950)
 * @return nothing or error string
 */
MRMESH_API Expected<void> zlibDecompressStream( std::istream& in, std::ostream& out, bool rawDeflate = false );

} // namespace MR
