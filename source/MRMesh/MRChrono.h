#pragma once

#include "MRMeshFwd.h"

#include <ctime>

namespace MR
{

// A threadsafe equivalent for `std::localtime()`. Returns null on failure.
[[nodiscard]] MRMESH_API std::optional<std::tm> Localtime( std::time_t time );

}
