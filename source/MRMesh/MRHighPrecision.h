#pragma once

#include "MRMeshFwd.h"
#include "MRFastInt.h"
#include "MRFastInt128.h"
#include <MRPch/MRBindingMacros.h>

namespace MR
{

/// \defgroup HighPrecisionGroup High Precision
/// \ingroup MathGroup
/// \{

// no bindings since no operator << and no sqrt for FastInt128
#if !defined MR_PARSING_FOR_ANY_BINDINGS && !defined MR_COMPILING_ANY_BINDINGS
using Vector2i128fast = Vector2<FastInt128>;
using Vector3i128fast = Vector3<FastInt128>;
#endif

/// \}

} // namespace MR
