#pragma once

#include "exports.h"
#include "MRUnits.h"

// This abstracts away the UI measurement unit configuration. Internally this uses `setDefaultUnitParams()`.

namespace MR::UnitSettings
{

// If true, the change will be written to `getViewerInstance().getViewerSettingsManager()`.
enum class WriteToSettings { no, yes };

// Load all settings from `getViewerInstance().getViewerSettingsManager()`.
MRVIEWER_API void loadFromViewerSettings();


// Common:

// True: `0.1`, false: `.1`.
[[nodiscard]] MRVIEWER_API bool getShowLeadingZero();
MRVIEWER_API void setShowLeadingZero( bool show, WriteToSettings writeToSettings = WriteToSettings::yes );

// Can be '\0' to indicate no separator.
// `fractional == true` means to the right of the fractional point, if any.
[[nodiscard]] MRVIEWER_API char getThousandsSeparator();
void setThousandsSeparator( char ch, WriteToSettings writeToSettings = WriteToSettings::yes );

// Length:

// In addition to length, this also controls the units for speed, area, volume, etc.
// This can be null to indicate "no unit".
// If `setPreferredLeadingZero == true`, will call `setShowLeadingZero()` to match this unit (currently inches = false, everything else = true).
[[nodiscard]] MRVIEWER_API std::optional<LengthUnit> getUiLengthUnit();
MRVIEWER_API void setUiLengthUnit( std::optional<LengthUnit> unit, bool setPreferredLeadingZero, WriteToSettings writeToSettings = WriteToSettings::yes );

// Angle:

[[nodiscard]] MRVIEWER_API DegreesMode getDegreesMode();
MRVIEWER_API void setDegreesMode( DegreesMode mode, bool setPreferredPrecision, WriteToSettings writeToSettings = WriteToSettings::yes );

// Precision:

// Whether this means total number of digits or the number of digits after the decimal point depends
//   on another setting (`getDefaultUnitParams().style`) that's currently not exposed in this file.
[[nodiscard]] MRVIEWER_API int getUiLengthPrecision();
MRVIEWER_API void setUiLengthPrecision( int precision, WriteToSettings writeToSettings = WriteToSettings::yes );

[[nodiscard]] MRVIEWER_API int getUiAnglePrecision();
MRVIEWER_API void setUiAnglePrecision( int precision, WriteToSettings writeToSettings = WriteToSettings::yes );

}
