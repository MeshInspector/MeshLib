#pragma once

#include "MRMeshFwd.h"
#include "MRFastInt128.h"
#include <MRPch/MRBindingMacros.h>

#if defined(__APPLE__) && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <boost/multiprecision/cpp_int.hpp>

#if defined(__APPLE__) && defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace MR
{

/// \defgroup HighPrecisionGroup High Precision
/// \ingroup MathGroup
/// \{

using Int128 = boost::multiprecision::int128_t;
using Int256 = boost::multiprecision::int256_t;

using Vector2i128 = Vector2<Int128>;
using Vector3i128 = Vector3<Int128>;

// no bindings since no operator << and no sqrt for FastInt128
#if !defined MR_PARSING_FOR_ANY_BINDINGS && !defined MR_COMPILING_ANY_BINDINGS
using Vector2i128fast = Vector2<FastInt128>;
using Vector3i128fast = Vector3<FastInt128>;
#endif

using Vector2i256 = Vector3<Int256>;
using Vector3i256 = Vector3<Int256>;


/// \}

} // namespace MR
