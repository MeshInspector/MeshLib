#pragma once

#include "exports.h"
#include "MRUnits.h"

// This abstracts away the UI measurement unit configuration. Internally this uses `setDefaultUnitParams()`.

namespace MR::UnitSettings
{

// Common:

// True: `0.1`, false: `.1`.
[[nodiscard]] MRVIEWER_API bool getShowLeadingZero();
MRVIEWER_API void setShowLeadingZero( bool show );

// Can be '\0' to indicate no separator.
// `fractional == true` means to the right of the fractional point, if any.
[[nodiscard]] MRVIEWER_API char getThousandsSeparator( bool fractional );
void setThousandsSeparator( char ch, bool fractional );

// Length:

// In addition to length, this also controls the units for speed, area, volume, etc.
// This can be null to indicate "no unit".
[[nodiscard]] MRVIEWER_API std::optional<LengthUnit> getUiLengthUnit();
MRVIEWER_API void setUiLengthUnit( std::optional<LengthUnit> unit );

// Angle:

[[nodiscard]] MRVIEWER_API DegreesMode getDegreesMode();
// This also calls `setUiAnglePrecision()` to set a default precision depending on the mode.
MRVIEWER_API void setDegreesMode( DegreesMode mode );

// Precision:

// Whether this means total number of digits or the number of digits after the decimal point depends
//   on another setting (`getDefaultUnitParams().style`) that's currently not exposed in this file.
[[nodiscard]] MRVIEWER_API int getUiLengthPrecision();
MRVIEWER_API void setUiLengthPrecision( int precision );

[[nodiscard]] MRVIEWER_API int getUiAnglePrecision();
MRVIEWER_API void setUiAnglePrecision( int precision );

}
