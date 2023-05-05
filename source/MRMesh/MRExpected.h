#pragma once

#include "MRMeshFwd.h"
#include <tl/expected.hpp>
#include <string>

namespace MR
{

/// return type for a void function that can produce an error string
using VoidOrErrStr = tl::expected<void, std::string>;

} //namespace MR
