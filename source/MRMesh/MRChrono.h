#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"

#include <ctime>
#include <optional>

namespace MR
{

// A threadsafe equivalent for `std::localtime()`. Returns null on failure.
[[nodiscard]] MRMESH_API MR_BIND_IGNORE std::optional<std::tm> Localtime( std::time_t time );

// Same, but returns a struct full of zeroes on error.
[[nodiscard]] MRMESH_API MR_BIND_IGNORE std::tm LocaltimeOrZero( std::time_t time );

}
