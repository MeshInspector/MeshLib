#pragma once

#include "MRMesh/MRMacros.h"
#include "MRViewer/exports.h"
#include "MRViewer/MRVectorTraits.h"

#include <cassert>
#include <optional>
#include <string>
#include <variant>

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
    meters,
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
    milliseconds,
    _count [[maybe_unused]],
};

// Measurement units for movement speed.
enum class MovementSpeedUnit
{
    mmPerSecond,
    metersPerSecond,
    inchesPerSecond,
    _count [[maybe_unused]],
};

// Measurement units for surface area.
enum class AreaUnit
{
    mm2,
    meters2,
    inches2,
    _count [[maybe_unused]],
};

// Measurement units for body volume.
enum class VolumeUnit
{
    mm3,
    meters3,
    inches3,
    _count [[maybe_unused]],
};

// Measurement units for 1/length.
enum class InvLengthUnit
{
    inv_mm, // mm^-1
    inv_meters, // meters^-1
    inv_inches, // inches^-1
    _count [[maybe_unused]],
};

// A list of all unit enums, for internal use.
#define DETAIL_MR_UNIT_ENUMS(X) X(NoUnit) X(LengthUnit) X(AngleUnit) X(PixelSizeUnit) X(RatioUnit) X(TimeUnit) X(MovementSpeedUnit) X(AreaUnit) X(VolumeUnit) X(InvLengthUnit)

// All supported value types for `valueToString()`.
// Not using `__VA_OPT__(,)` here to support legacy MSVC preprocessor.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#define DETAIL_MR_UNIT_VALUE_TYPES(X, ...) \
    X(float       ,__VA_ARGS__) X(double              ,__VA_ARGS__) X(long double ,__VA_ARGS__) \
    X(signed char ,__VA_ARGS__) X(unsigned char       ,__VA_ARGS__) \
    X(short       ,__VA_ARGS__) X(unsigned short      ,__VA_ARGS__) \
    X(int         ,__VA_ARGS__) X(unsigned int        ,__VA_ARGS__) \
    X(long        ,__VA_ARGS__) X(unsigned long       ,__VA_ARGS__) \
    X(long long   ,__VA_ARGS__) X(unsigned long long  ,__VA_ARGS__)
#ifdef __clang__
#pragma clang diagnostic pop
#endif

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
// This version also returns true if `a` or `b` is null.
template <UnitEnum E>
[[nodiscard]] bool unitsAreEquivalent( const std::optional<E> &a, const std::optional<E> &b )
{
    return !a || !b || unitsAreEquivalent( *a, *b );
}

namespace detail::Units
{
    struct Empty {};

    template <typename T>
    concept Scalar = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

    template <typename T>
    using MakeFloatingPoint = std::conditional_t<std::is_integral_v<typename VectorTraits<T>::BaseType>, typename VectorTraits<T>::template ChangeBaseType<float>, T>;
}

// Converts `value` from unit `from` to unit `to`. `value` is a scalar of a Vector2/3/4 or ImVec2/4 of them.
// The return type matches `T` if it's not integral. If it's integral, its element type type is changed to `float`.
// Returns min/max floating-point values unchanged.
template <UnitEnum E, typename T>
[[nodiscard]] detail::Units::MakeFloatingPoint<T> convertUnits( E from, E to, const T& value )
{
    using ReturnType = detail::Units::MakeFloatingPoint<T>;

    bool needConversion = !unitsAreEquivalent( from, to );

    if constexpr ( std::is_same_v<T, ReturnType> )
    {
        if ( !needConversion )
            return value;
    }

    ReturnType ret{};

    for ( int i = 0; i < VectorTraits<T>::size; i++ )
    {
        auto& target = VectorTraits<ReturnType>::getElem( i, ret ) = (typename VectorTraits<ReturnType>::BaseType) VectorTraits<T>::getElem( i, value );

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

// This version is a no-op if `from` or `to` is null.
template <UnitEnum E, typename T>
[[nodiscard]] detail::Units::MakeFloatingPoint<T> convertUnits( const std::optional<E> &from, const std::optional<E> &to, const T& value )
{
    if ( from && to )
        return convertUnits( *from, *to, value );
    else
        return detail::Units::MakeFloatingPoint<T>( value );
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
