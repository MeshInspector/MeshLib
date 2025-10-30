#pragma once

#include "MRViewer/exports.h"
#include "MRViewer/MRVectorTraits.h"
#include "MRMesh/MRUnitInfo.h"

#include <string>
#include <variant>

// Read the manual at: docs/measurement_units.md

namespace MR
{

namespace detail::Units
{
    struct Empty {};
}

template <UnitEnum E>
struct UnitToStringParams;

// Returns the default parameters for converting a specific unit type to a string.
// You can modify those with `setDefaultUnitParams()`.
template <UnitEnum E>
[[nodiscard]] const UnitToStringParams<E>& getDefaultUnitParams();

#define MR_X(E) extern template MRVIEWER_API const UnitToStringParams<E>& getDefaultUnitParams();
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X

// Modifies the default parameters for converting a specific unit type to a string.
template <UnitEnum E>
void setDefaultUnitParams( const UnitToStringParams<E>& newParams );

#define MR_X(E) extern template MRVIEWER_API void setDefaultUnitParams( const UnitToStringParams<E>& newParams );
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X

enum class NumberStyle
{
    normal, // Like %f.
    distributePrecision, // Like %f, but the precision digits are spread across both decimal and integral parts.
    exponential, // Like %e.
    maybeExponential, // Like %g.
};

// This controls how the degrees are printed.
enum class DegreesMode
{
    degrees, // Fractional degrees.
    degreesMinutes, // Integral degrees, fractional arcminutes.
    degreesMinutesSeconds, // Integral degrees and minutes, fractional arcseconds.
    _count [[maybe_unused]],
};
[[nodiscard]] MRVIEWER_API std::string_view toString( DegreesMode mode );

enum class ZeroMode
{
    asIs, // Print as is.
    alwaysPositive, // Treat negative zero as positive zero.
    alwaysNegative, // Treat positive zero as negative zero.
};

// Controls how a value with a unit is converted to a string.
template <UnitEnum E>
struct UnitToStringParams
{
    // The resulting string is wrapped in this.
    // Do NOT use this for custom unit suffixes! Add them as actual units instead.
    std::string_view decorationFormatString = "{}";

    // --- Units:

    // The measurement unit of the input value. If null, no conversion is performed.
    std::optional<E> sourceUnit = getDefaultUnitParams<E>().sourceUnit;
    // The measurement unit of the resulting string. If null, no conversion is performed, and the unit name is taken from `sourceUnit` if any.
    std::optional<E> targetUnit = getDefaultUnitParams<E>().targetUnit;

    // Whether to show the unit suffix.
    bool unitSuffix = getDefaultUnitParams<E>().unitSuffix;

    // --- Precision:

    // The output style. (Scientific notation or not, fixed-precision or not.)
    NumberStyle style = getDefaultUnitParams<E>().style;

    // How many digits of precision.
    int precision = getDefaultUnitParams<E>().precision;

    // --- Other:

    // Show the `+` sign on positive numbers.
    bool plusSign = getDefaultUnitParams<E>().plusSign;

    // How to deal with zeroes.
    ZeroMode zeroMode = getDefaultUnitParams<E>().zeroMode;

    // Use a pretty Unicode minus sign instead of the ASCII `-`.
    bool unicodeMinusSign = getDefaultUnitParams<E>().unicodeMinusSign;

    // If non-zero, this character is inserted between every three digits to the left of the decimal point.
    char thousandsSeparator = getDefaultUnitParams<E>().thousandsSeparator;
    // If non-zero, this character is inserted between every three digits to the right of the decimal point.
    char thousandsSeparatorFrac = getDefaultUnitParams<E>().thousandsSeparatorFrac;

    // If false, remove zero before the fractional point (`.5` instead of `0.5`).
    bool leadingZero = getDefaultUnitParams<E>().leadingZero;

    // Remove trailing zeroes after the fractional point. If the point becomes the last symbol, remove the point too.
    bool stripTrailingZeroes = getDefaultUnitParams<E>().stripTrailingZeroes;

    // When printing degrees, this lets you display arcminutes and possibly arcseconds. Ignored for everything else.
    std::conditional_t<std::is_same_v<E, AngleUnit>, DegreesMode, detail::Units::Empty> degreesMode = getDefaultUnitParams<E>().degreesMode;

    // If you add new fields there, update the initializer for `defaultUnitToStringParams` in `MRUnits.cpp`.

    friend bool operator==( const UnitToStringParams&, const UnitToStringParams& ) = default;
};

// The `std::variant` of `UnitToStringParams<E>` for all known `E`s (unit kinds).
using VarUnitToStringParams = std::variant<
    #define MR_X(E) , UnitToStringParams<E>
    MR_TRIM_LEADING_COMMA(DETAIL_MR_UNIT_ENUMS(MR_X))
    #undef MR_X
>;

// Converts value to a string, possibly converting it to a different unit.
// By default, length is kept as is, while angles are converted from radians to the current UI unit.
template <UnitEnum E, detail::Units::Scalar T>
[[nodiscard]] MRVIEWER_API std::string valueToString( T value, const UnitToStringParams<E>& params = getDefaultUnitParams<E>() );

#define MR_Y(T, E) extern template MRVIEWER_API std::string valueToString<E, T>( T value, const UnitToStringParams<E>& params );
#define MR_X(E) DETAIL_MR_UNIT_VALUE_TYPES(MR_Y, E)
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X
#undef MR_Y

// This overload lets you select the unit kind at runtime.
template <detail::Units::Scalar T>
[[nodiscard]] MRVIEWER_API std::string valueToString( T value, const VarUnitToStringParams& params );

#define MR_X(T, unused) extern template MRVIEWER_API std::string valueToString( T value, const VarUnitToStringParams& params );
DETAIL_MR_UNIT_VALUE_TYPES(MR_X,)
#undef MR_X

// Guesses the number of digits of precision for fixed-point formatting of `value`.
// Mostly for internal use.
template <detail::Units::Scalar T>
[[nodiscard]] MRVIEWER_API int guessPrecision( T value );

// Guesses the number of digits of precision for fixed-point formatting of the min-max range.
// If `min >= max`, always returns zero. Ignores min and/or max if they are the smallest of the largest representable value respectively.
// Mostly for internal use.
template <detail::Units::Scalar T>
[[nodiscard]] MRVIEWER_API int guessPrecision( T min, T max );

// Same but for vectors.
template <typename T>
requires (VectorTraits<T>::size > 1 && detail::Units::Scalar<typename VectorTraits<T>::BaseType>)
[[nodiscard]] int guessPrecision( T value )
{
    int ret = 0;
    for ( int i = 0; i < VectorTraits<T>::size; i++ )
        ret = std::max( ret, guessPrecision( VectorTraits<T>::getElem( i, value ) ) );
    return ret;
}
template <typename T>
requires (VectorTraits<T>::size > 1 && detail::Units::Scalar<typename VectorTraits<T>::BaseType>)
[[nodiscard]] int guessPrecision( T min, T max )
{
    int ret = 0;
    for ( int i = 0; i < VectorTraits<T>::size; i++ )
        ret = std::max( ret, guessPrecision( VectorTraits<T>::getElem( i, min ), VectorTraits<T>::getElem( i, max ) ) );
    return ret;
}

#define MR_X(T, unused) \
    extern template MRVIEWER_API int guessPrecision( T value ); \
    extern template MRVIEWER_API int guessPrecision( T min, T max );
DETAIL_MR_UNIT_VALUE_TYPES(MR_X,)
#undef MR_X

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

}
