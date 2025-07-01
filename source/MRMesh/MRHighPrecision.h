#pragma once

#include "MRMeshFwd.h"
#include "MRFastInt128.h"
#include <boost/multiprecision/cpp_int.hpp>

namespace MR
{

/// \defgroup HighPrecisionGroup High Precision
/// \ingroup MathGroup
/// \{

using Int64 = long long;
static_assert( sizeof( Int64 ) == 8 );
using Int128 = boost::multiprecision::int128_t;
using Int256 = boost::multiprecision::int256_t;

using Vector2i64 = Vector2<Int64>;
using Vector3i64 = Vector3<Int64>;

using Vector2i128 = Vector2<Int128>;
using Vector3i128 = Vector3<Int128>;

using Vector2i128fast = Vector2<FastInt128>;
using Vector3i128fast = Vector3<FastInt128>;

using Vector2i256 = Vector3<Int256>;
using Vector3i256 = Vector3<Int256>;


/// \}

} // namespace MR
