#pragma once

#include "MRMeshFwd.h"
#include <boost/multiprecision/cpp_int.hpp>

namespace MR
{

using HighPrecisionInt = boost::multiprecision::checked_int128_t;

using Vector2hp = Vector2<HighPrecisionInt>;
using Vector3hp = Vector3<HighPrecisionInt>;

} //namespace MR
