#pragma once

#include "MRMeshFwd.h"
#include "MRFastInt128.h"
#include <MRPch/MRBindingMacros.h>
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

// ignored since no operator << for FastInt128
using Vector2i128fast MR_BIND_IGNORE = Vector2<FastInt128>;
using Vector3i128fast MR_BIND_IGNORE = Vector3<FastInt128>;

using Vector2i256 = Vector3<Int256>;
using Vector3i256 = Vector3<Int256>;


/// \}

} // namespace MR
