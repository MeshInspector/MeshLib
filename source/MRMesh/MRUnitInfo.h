#pragma once

#include "MRVectorTraits.h"
#include "MRMacros.h"
#include "MREnums.h"

#include <cassert>
#include <optional>

namespace MR
{

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

// ignore for bindings to prevent GCC14 error: undefined symbol: _ZN2MR11getUnitInfoITkNS_8UnitEnumENS_8TimeUnitEEERKNS_8UnitInfoET_
#define MR_X(E) template <> [[nodiscard]] MRMESH_API MR_BIND_IGNORE const UnitInfo& getUnitInfo( E unit );
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

} //namespace MR
