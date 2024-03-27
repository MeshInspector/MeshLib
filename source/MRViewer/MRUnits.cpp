#include "MRUnits.h"

#include "MRMesh/MRConstants.h"
#include "MRPch/MRFmt.h"

namespace MR
{

template <UnitEnum E>
static constinit UnitToStringParams<E> defaultUnitToStringParams = []{
    if constexpr ( std::is_same_v<E, NoUnit> )
    {
        return UnitToStringParams<NoUnit>{
            .sourceUnit = std::nullopt,
            .targetUnit = {},
            .unitSuffix = false,
            .style = NumberStyle::normal,
            .precision = 3,
            .unicodeMinusSign = true,
            .thousandsSeparator = ' ',
            .leadingZero = true,
            .stripTrailingZeroes = true,
            .degreesMode = {},
        };
    }
    else if constexpr ( std::is_same_v<E, LengthUnit> )
    {
        return UnitToStringParams<LengthUnit>{
            .sourceUnit = std::nullopt,
            .targetUnit = LengthUnit::mm,
            .unitSuffix = true,
            .style = NumberStyle::normal,
            .precision = 3,
            .unicodeMinusSign = true,
            .thousandsSeparator = ' ',
            .leadingZero = true,
            .stripTrailingZeroes = true,
            .degreesMode = {},
        };
    }
    else if constexpr ( std::is_same_v<E, AngleUnit> )
    {
        return UnitToStringParams<AngleUnit>{
            .sourceUnit = AngleUnit::radians,
            .targetUnit = AngleUnit::degrees,
            .unitSuffix = true,
            .style = NumberStyle::fixed,
            .precision = 1,
            .unicodeMinusSign = true,
            .thousandsSeparator = 0,
            .leadingZero = true,
            .stripTrailingZeroes = true,
            .degreesMode = {},
        };
    }
    else if constexpr ( std::is_same_v<E, PixelSizeUnit> )
    {
        return UnitToStringParams<PixelSizeUnit>{
            .sourceUnit = std::nullopt,
            .targetUnit = PixelSizeUnit::pixels,
            .unitSuffix = true,
            .style = NumberStyle::normal,
            .precision = 2,
            .unicodeMinusSign = true,
            .thousandsSeparator = ' ',
            .leadingZero = true,
            .stripTrailingZeroes = true,
            .degreesMode = {},
        };
    }
    else if constexpr ( std::is_same_v<E, RatioUnit> )
    {
        return UnitToStringParams<RatioUnit>{
            .sourceUnit = RatioUnit::factor,
            .targetUnit = RatioUnit::percents,
            .unitSuffix = true,
            .style = NumberStyle::fixed,
            .precision = 1,
            .unicodeMinusSign = true,
            .thousandsSeparator = ' ',
            .leadingZero = true,
            .stripTrailingZeroes = true,
            .degreesMode = {},
        };
    }
    else if constexpr ( std::is_same_v<E, TimeUnit> )
    {
        return UnitToStringParams<TimeUnit>{
            .sourceUnit = TimeUnit::seconds,
            .targetUnit = TimeUnit::seconds,
            .unitSuffix = true,
            .style = NumberStyle::fixed,
            .precision = 1,
            .unicodeMinusSign = true,
            .thousandsSeparator = ' ',
            .leadingZero = true,
            .stripTrailingZeroes = true,
            .degreesMode = {},
        };
    }
    else
    {
        static_assert( dependent_false<E>, "Unknown measurement unit type." );
    }
}();

template <>
const UnitInfo& getUnitInfo( NoUnit noUnit )
{
    assert( false );
    (void)noUnit;
    static const UnitInfo ret{};
    return ret;
}
template <>
const UnitInfo& getUnitInfo( LengthUnit length )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = "Mm", .unitSuffix = " mm" },
        { .conversionFactor = 25.4f, .prettyName = "Inches", .unitSuffix = " in"/* or "\"" */ },
    };
    static_assert( std::extent_v<decltype( ret )> == int( LengthUnit::_count ) );
    return ret[int( length )];
}
template <>
const UnitInfo& getUnitInfo( AngleUnit angle )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = "Radians", .unitSuffix = " radians" },
        { .conversionFactor = PI_F/180.f, .prettyName = "Degrees", .unitSuffix = "\xC2\xB0" }, // U+00B0 DEGREE SIGN
    };
    static_assert( std::extent_v<decltype( ret )> == int( AngleUnit::_count ) );
    return ret[int( angle )];
}
template <>
const UnitInfo& getUnitInfo( PixelSizeUnit screenSize )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = "Pixels", .unitSuffix = " px" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( PixelSizeUnit::_count ) );
    return ret[int( screenSize )];
}
template <>
const UnitInfo& getUnitInfo( RatioUnit ratio )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = "Factor", .unitSuffix = " x" },
        { .conversionFactor = 100, .prettyName = "Percents", .unitSuffix = " %" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( RatioUnit::_count ) );
    return ret[int( ratio )];
}
template <>
const UnitInfo& getUnitInfo( TimeUnit time )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = "Seconds", .unitSuffix = " sec" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( TimeUnit::_count ) );
    return ret[int( time )];
}

template <UnitEnum E>
const UnitToStringParams<E>& getDefaultUnitParams()
{
    return defaultUnitToStringParams<E>;
}

#define MR_X(E) template MRVIEWER_API const UnitToStringParams<E>& getDefaultUnitParams();
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X

template <UnitEnum E>
void setDefaultUnitParams( const UnitToStringParams<E>& newParams )
{
    defaultUnitToStringParams<E> = newParams;
}

#define MR_X(E) template void setDefaultUnitParams( const UnitToStringParams<E>& newParams );
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X

template <UnitEnum E, typename T>
static std::string valueToStringImpl( T value, const UnitToStringParams<E>& params )
{
    // If `str` starts with ASCII minus minus, and `params.unicodeMinusSign` is set, replace it with a Unicode minus sign.
    auto adjustMinusSign = [&]( std::string& str )
    {
        if ( params.unicodeMinusSign && str.starts_with( '-' ) )
        {
            // U+2212 MINUS SIGN.
            str[0] = '\xe2';
            str.insert( str.begin() + 1, { '\x88', '\x92' } );
        }
    };

    std::string_view unitSuffix;
    if ( params.unitSuffix )
        unitSuffix = getUnitInfo( params.targetUnit ).unitSuffix;
    std::string ret;

    // Handle arcseconds/arcminutes.
    // Write all but the last component to `ret. Set `value` and `unixSuffix` to the last component.
    if constexpr ( std::is_same_v<E, AngleUnit> && std::is_floating_point_v<T> )
    {
        if ( params.targetUnit == AngleUnit::degrees )
        {
            if ( params.degreesMode == DegreesMode::degreesMinutes || params.degreesMode == DegreesMode::degreesMinutesSeconds )
            {
                bool negative = value < 0;
                if ( negative )
                    value = -value;

                float wholeDegrees = 0;
                float minutes = std::modf( value, &wholeDegrees ) * 60;
                if ( negative )
                    wholeDegrees = -wholeDegrees;

                ret = fmt::format( "{:.0f}{}", wholeDegrees, getUnitInfo( AngleUnit::degrees ).unitSuffix );
                adjustMinusSign( ret );

                if ( params.degreesMode == DegreesMode::degreesMinutesSeconds )
                {
                    float wholeMinutes = 0;
                    float seconds = std::modf( minutes, &wholeMinutes ) * 60;

                    ret += fmt::format( "{}'", wholeMinutes );

                    value = seconds;
                    unitSuffix = "\"";
                }
                else
                {
                    value = minutes;
                    unitSuffix = "'";
                }
            }
        }
    }

    auto formatValue = [&]( T value, int precision ) -> std::string
    {
        if constexpr ( std::is_floating_point_v<T> )
        {
            if ( params.style == NumberStyle::maybeScientific )
                return fmt::format( "{:.{}g}", value, precision );
            else if ( params.style == NumberStyle::scientific )
                return fmt::format( "{:.{}e}", value, precision );
            else
                return fmt::format( "{:.{}f}", value, precision );
        }
        else
        {
            (void)precision;
            return fmt::format( "{}", value );
        }
    };

    // Calculate precision after the decimal point.
    int fracPrecision = std::is_floating_point_v<T> ? params.precision : 0;
    if ( params.style == NumberStyle::normal && fracPrecision > 0 )
    {
        int intDigits = 0;

        std::string tmp = formatValue( value, params.precision );
        std::size_t dot = tmp.find( '.' );
        if ( dot != std::string::npos )
            intDigits = int( dot ) - ( tmp.front() == '-' );

        fracPrecision -= intDigits;
    }
    if ( fracPrecision < 0 )
        fracPrecision = 0;

    { // Format the value.
        std::string formattedValue = formatValue( value, fracPrecision );

        // Remove the leading zero.
        if constexpr ( std::is_floating_point_v<T> )
        {
            if ( !params.leadingZero  )
            {
                if ( formattedValue.starts_with( "0." ) )
                    formattedValue.erase( formattedValue.begin() );
                else if ( formattedValue.starts_with( "-0." ) )
                    formattedValue.erase( formattedValue.begin() + 1 );
            }
        }

        // Add the thousands separator.
        if ( params.thousandsSeparator )
        {
            auto pos = formattedValue.find( '.' );
            if ( pos == std::string::npos )
                pos = formattedValue.size();

            while ( pos > 3 && std::isdigit( (unsigned char)formattedValue[pos - 4] ) )
            {
                pos -= 3;
                formattedValue.insert( formattedValue.begin() + pos, params.thousandsSeparator );
            }
        }

        // Remove the trailing zeroes.
        if constexpr ( std::is_floating_point_v<T> )
        {
            if ( params.stripTrailingZeroes && formattedValue.find( '.' ) != std::string::npos && formattedValue.find( 'e' ) == std::string::npos )
            {
                bool strippedAny = false;
                while ( formattedValue.ends_with( '0' ) )
                {
                    strippedAny = true;
                    formattedValue.pop_back();
                }

                if ( strippedAny && formattedValue.ends_with( '.' ) )
                    formattedValue.pop_back();
            }
        }

        adjustMinusSign( formattedValue );

        ret += formattedValue;
    }

    ret += unitSuffix;

    return ret;
}

template <UnitEnum E, typename T>
requires (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
std::string valueToString( T value, const UnitToStringParams<E>& params )
{
    // Convert to the target unit.
    if ( unitsAreEquivalent( params.sourceUnit.value_or( params.targetUnit ), params.targetUnit ) )
    {
        // This can be integral or floating-point.
        return valueToStringImpl( value, params );
    }
    else
    {
        // This is always integral.
        return valueToStringImpl( convertUnits( *params.sourceUnit, params.targetUnit, value ), params );
    }
}

#define MR_Y(T, E) template std::string valueToString<E, T>( T value, const UnitToStringParams<E>& params );
#define MR_X(E) DETAIL_MR_UNIT_VALUE_TYPES(MR_Y, E)
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X
#undef MR_Y

template <typename T>
requires (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
int guessPrecision( T value )
{
    if constexpr ( std::is_integral_v<T> )
    {
        (void)value;
        return 0;
    }
    else
    {
        if ( !std::isnormal( value ) )
            return 0; // Reject non-finite numbers and zeroes.

        if ( value < 0 )
            value = -value;

        if ( value >= 1 )
            return 0;

        std::string str = fmt::format( "{:.{}f}", value, std::numeric_limits<T>::max_digits10 );

        auto pos = str.find_first_not_of( "0." );
        if ( pos == std::string::npos )
            return 0; // Should be impossible?

        // pos - 2 == the number of leading zeroes in the fractional part, then +1 to show the first digit.

        if ( pos == 0 )
            return 0;
        return pos - 1;
    }
}

template <typename T>
requires (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
int guessPrecision( T min, T max )
{
    if constexpr ( std::is_integral_v<T> )
    {
        (void)min;
        (void)max;
        return 0;
    }
    else
    {
        if ( !( min < max ) )
            return 0;

        bool haveMin = min > std::numeric_limits<T>::lowest();
        bool haveMax = max < std::numeric_limits<T>::max();

        if ( !haveMin && !haveMax )
            return 0;

        if ( haveMin && !haveMax )
            return guessPrecision( min );
        if ( !haveMin && haveMax )
            return guessPrecision( max );

        int a = guessPrecision( min );
        int b = guessPrecision( max );

        if ( a == b && min * 2 >= max )
            return a + 1;

        return std::max( a, b );
    }
}

#define MR_X(T) \
    template int guessPrecision( T value ); \
    template int guessPrecision( T min, T max );
DETAIL_MR_UNIT_VALUE_TYPES(MR_X)
#undef MR_X

}
