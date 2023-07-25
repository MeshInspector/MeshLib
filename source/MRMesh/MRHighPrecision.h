#pragma once

#include "MRMeshFwd.h"
#include <boost/multiprecision/cpp_int.hpp>

namespace MR
{

/// \defgroup HighPrecisionGroup High Precision
/// \ingroup MathGroup
/// \{

using HighPrecisionInt = boost::multiprecision::checked_int128_t;

using Vector2hp = Vector2<HighPrecisionInt>;
using Vector3hp = Vector3<HighPrecisionInt>;
using Vector4hp = Vector4<HighPrecisionInt>;
using Matrix3hp = Matrix3<HighPrecisionInt>;
using Matrix4hp = Matrix4<HighPrecisionInt>;


/// \}

} // namespace MR
