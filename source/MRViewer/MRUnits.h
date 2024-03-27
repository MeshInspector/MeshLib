#pragma once

#include "MRViewer/exports.h"
#include "MRViewer/MRVectorTraits.h"

#include <cassert>
#include <optional>
#include <string>

// Read the manual at: docs/measurement_units.md

namespace MR
{

// A stub measurement unit representing no unit.
enum class NoUnit
{
    _count [[maybe_unused]]
};

// Measurement units of length.
enum class LengthUnit
{
    mm,
    inches,
    _count [[maybe_unused]],
};

// Measurement units of angle.
enum class AngleUnit
{
    radians,
    degrees,
    _count [[maybe_unused]],
};

// Measurement units of screen sizes.
enum class PixelSizeUnit
{
    pixels,
    _count [[maybe_unused]],
};

// Measurement units for factors / ratios.
enum class RatioUnit
{
    factor, // 0..1 x
    percents, // 0..100 %
    _count [[maybe_unused]],
};

// Measurement units for time.
enum class TimeUnit
{
    seconds,
    _count [[maybe_unused]],
};

// A list of all unit enums, for internal use.
#define DETAIL_MR_UNIT_ENUMS(X) X(NoUnit) X(LengthUnit) X(AngleUnit) X(PixelSizeUnit) X(RatioUnit) X(TimeUnit)

// All supported value types for `valueToString()`.
#define DETAIL_MR_UNIT_VALUE_TYPES(X, ...) \
    X(float      , __VA_ARGS__) X(double        , __VA_ARGS__) X(long double, __VA_ARGS__) \
    X(signed char, __VA_ARGS__) X(unsigned char , __VA_ARGS__) \
    X(short      , __VA_ARGS__) X(unsigned short, __VA_ARGS__) \
    X(int        , __VA_ARGS__) X(unsigned int  , __VA_ARGS__) \
    X(long       , __VA_ARGS__) X(unsigned long , __VA_ARGS__)

// Whether `E` is one of the unit enums: NoUnit, LengthUnit, AngleUnit, ...
template <typename T>
concept UnitEnum =
    #define MR_X(E) || std::same_as<T, E>
    true DETAIL_MR_UNIT_ENUMS(MR_X);
    #undef MR_X

// ---

// Information about a single measurement unit.
struct UnitInfo
{
    // This is used to convert between units.
    // To convert from A to B, multiply by A's factor and divide by B's.
    float conversionFactor = 1;

    std::string_view prettyName;

    // The short unit name that's placed after values.
    // This may or may not start with a space.
    std::string_view unitSuffix;
};

// Returns information about a single measurement unit.
template <UnitEnum E>
[[nodiscard]] const UnitInfo& getUnitInfo( E unit ) = delete;

#define MR_X(E) template <> [[nodiscard]] MRVIEWER_API const UnitInfo& getUnitInfo( E unit );
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X

// Returns true if converting a value between units `a` and `b` doesn't change its value.
template <UnitEnum E>
[[nodiscard]] bool unitsAreEquivalent( E a, E b )
{
    return a == b || getUnitInfo( a ).conversionFactor == getUnitInfo( b ).conversionFactor;
}

// Converts `value` from unit `from` to unit `to`. `value` is a scalar of a Vector2/3/4 or ImVec2/4 of them.
// The return type matches `T` if it's not integral. If it's integral, its element type type is changed to `float`.
// Returns min/max floating-point values unchanged.
template <UnitEnum E, typename T, typename ReturnType = std::conditional_t<std::is_integral_v<typename VectorTraits<T>::BaseType>, typename VectorTraits<T>::template ChangeBaseType<float>, T>>
[[nodiscard]] ReturnType convertUnits( E from, E to, const T& value )
{
    bool needConversion = !unitsAreEquivalent( from, to );

    if constexpr ( std::is_same_v<T, ReturnType> )
    {
        if ( !needConversion )
            return value;
    }

    ReturnType ret{};

    for ( int i = 0; i < VectorTraits<T>::size; i++ )
    {
        auto& target = VectorTraits<T>::getElem( i, ret ) = (typename VectorTraits<T>::BaseType) VectorTraits<T>::getElem( i, value );

        // Don't touch min/max floating-point values.
        bool needElemConversion = needConversion;
        if constexpr ( std::is_floating_point_v<typename VectorTraits<T>::BaseType> )
        {
            if ( needElemConversion &&
                (
                    target <= std::numeric_limits<typename VectorTraits<T>::BaseType>::lowest() ||
                    target >= std::numeric_limits<typename VectorTraits<T>::BaseType>::max()
                )
            )
                needElemConversion = false;
        }

        if ( needElemConversion )
            target = target * getUnitInfo( from ).conversionFactor / getUnitInfo( to ).conversionFactor;
    }

    return ret;
}

// ---

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
    normal, // Like %f. The precision digits are spread across both decimal and integral parts
    fixed, // Like %f, but the precision digits only affect the decimal part.
    scientific, // Like %e.
    maybeScientific, // Like %g.
};

// This controls how the degrees are printed.
enum class DegreesMode
{
    degrees, // Fractional degrees.
    degreesMinutes, // Integral degrees, fractional arcminutes.
    degreesMinutesSeconds, // Integral degrees and minutes, fractional arcseconds.
};

// How the trailing zeroes are stripped.
// All of this only applies if the number has a decimal point.
enum class TrailingZeroes
{
    keep, // Don't touch trailing zeroes.
    stripAndKeepOne, // Strip trailing zeroes, but if the last character is `.` after that, add one zero back.
    stripAll, // Strip trailing zeroes unconditionally.
};

namespace detail::Units
{
    struct Empty {};
}

// Controls how a value with a unit is converted to a string.
template <UnitEnum E>
struct UnitToStringParams
{
    // --- Units:

    // The measurement unit of the input.
    // If null, assumed to be the same as `targetUnit`, and no conversion is performed.
    // If not null, the value is converted from this unit to `targetUnit`.
    std::optional<E> sourceUnit = getDefaultUnitParams<E>().sourceUnit;

    // The measurement unit of the result.
    E targetUnit = getDefaultUnitParams<E>().targetUnit;

    // Whether to show the unit suffix.
    bool unitSuffix = getDefaultUnitParams<E>().unitSuffix;

    // --- Precision:

    // The output style. (Scientific notation or not, fixed-precision or not.)
    NumberStyle style = getDefaultUnitParams<E>().style;

    // How many digits of precision.
    int precision = getDefaultUnitParams<E>().precision;

    // --- Other:

    // Use a pretty Unicode minus sign instead of the ASCII `-`.
    bool unicodeMinusSign = getDefaultUnitParams<E>().unicodeMinusSign;

    // If non-zero, this character is inserted between every three digits.
    char thousandsSeparator = getDefaultUnitParams<E>().thousandsSeparator;

    // If false, remove zero before the fractional point (`.5` instead of `0.5`).
    bool leadingZero = getDefaultUnitParams<E>().leadingZero;

    // Remove trailing zeroes after the fractional point. If the point becomes the last symbol, remove the point too.
    bool stripTrailingZeroes = getDefaultUnitParams<E>().stripTrailingZeroes;

    // When printing degrees, this lets you display arcminutes and possibly arcseconds. Ignored for everything else.
    std::conditional_t<std::is_same_v<E, AngleUnit>, DegreesMode, detail::Units::Empty> degreesMode = getDefaultUnitParams<E>().degreesMode;

    // If you add new fields there, update the initializer for `defaultUnitToStringParams` in `MRUnits.cpp`.

    friend bool operator==( const UnitToStringParams&, const UnitToStringParams& ) = default;
};

// Converts value to a string, possibly converting it to a different unit.
// By default, length is kept as is, while angles are converted from radians to the current UI unit.
template <UnitEnum E, typename T>
requires std::is_arithmetic_v<T>
std::string valueToString( T value, const UnitToStringParams<E>& params = getDefaultUnitParams<E>() );

#define MR_Y(T, E) extern template MRVIEWER_API std::string valueToString<E, T>( T value, const UnitToStringParams<E>& params );
#define MR_X(E) DETAIL_MR_UNIT_VALUE_TYPES(MR_Y, E)
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X
#undef MR_Y

}
