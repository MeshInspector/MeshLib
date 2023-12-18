#pragma once

#include "MRExpected.h"

#include <iostream>

namespace MR
{

/// ...
VoidOrErrStr zlibCompressStream( std::istream& in, std::ostream& out, int level = -1 );

/// ...
VoidOrErrStr zlibDecompressStream( std::istream& in, std::ostream& out );

} // namespace MR
