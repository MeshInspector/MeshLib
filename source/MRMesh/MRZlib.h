#pragma once

#include "MRExpected.h"

#include <iostream>

namespace MR
{

/// ...
MRMESH_API VoidOrErrStr zlibCompressStream( std::istream& in, std::ostream& out, int level = -1 );

/// ...
MRMESH_API VoidOrErrStr zlibDecompressStream( std::istream& in, std::ostream& out );

} // namespace MR
