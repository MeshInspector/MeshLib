#pragma once

#include "exports.h"
#include "MRMesh/MRUnits.h"

namespace MR
{

// Generates a printf-style format string for `value`, for use with ImGui widgets.
// It has form "123.45 mm##%.6f" (the baked number, then `##` and some format string).
// The `##...` part isn't printed, but we need it when ctrl+clicking the number, to show the correct number of digits.
template <UnitEnum E, detail::Units::Scalar T>
[[nodiscard]] MRVIEWER_API std::string valueToImGuiFormatString( T value, const UnitToStringParams<E>& params = getDefaultUnitParams<E>() );

#define MR_Y(T, E) extern template MRVIEWER_API std::string valueToImGuiFormatString( T value, const UnitToStringParams<E>& params );
#define MR_X(E) DETAIL_MR_UNIT_VALUE_TYPES(MR_Y, E)
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X
#undef MR_Y

// This overload lets you select the unit kind at runtime.
template <detail::Units::Scalar T>
[[nodiscard]] MRVIEWER_API std::string valueToImGuiFormatString( T value, const VarUnitToStringParams& params );

#define MR_X(T, unused) extern template MRVIEWER_API std::string valueToImGuiFormatString( T value, const VarUnitToStringParams& params );
DETAIL_MR_UNIT_VALUE_TYPES(MR_X,)
#undef MR_X

} //namespace MR
