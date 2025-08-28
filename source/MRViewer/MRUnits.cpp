#include "MRUnits.h"

#include "MRMesh/MRConstants.h"
#include "MRMesh/MRString.h"
#include "MRPch/MRFmt.h"
#include <algorithm>

namespace MR
{

template <UnitEnum E>
static constinit UnitToStringParams<E> defaultUnitToStringParams = []{
    constexpr char commonThouSep = ' ', commonThouSepFrac = 0;

    if constexpr ( std::is_same_v<E, NoUnit> )
    {
        return UnitToStringParams<NoUnit>{
            .sourceUnit = std::nullopt,
            .targetUnit = {},
            .unitSuffix = false,
            .style = NumberStyle::normal,
            .precision = 3,
            .plusSign = false,
            .zeroMode = ZeroMode::alwaysPositive,
            .unicodeMinusSign = true,
            .thousandsSeparator = commonThouSep,
            .thousandsSeparatorFrac = commonThouSepFrac,
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
            .plusSign = false,
            .zeroMode = ZeroMode::alwaysPositive,
            .unicodeMinusSign = true,
            .thousandsSeparator = commonThouSep,
            .thousandsSeparatorFrac = commonThouSepFrac,
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
            .style = NumberStyle::normal,
            .precision = 1,
            .plusSign = false,
            .zeroMode = ZeroMode::alwaysPositive,
            .unicodeMinusSign = true,
            .thousandsSeparator = commonThouSep,
            .thousandsSeparatorFrac = commonThouSepFrac,
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
            .plusSign = false,
            .zeroMode = ZeroMode::alwaysPositive,
            .unicodeMinusSign = true,
            .thousandsSeparator = commonThouSep,
            .thousandsSeparatorFrac = commonThouSepFrac,
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
            .style = NumberStyle::normal,
            .precision = 3,
            .plusSign = false,
            .zeroMode = ZeroMode::alwaysPositive,
            .unicodeMinusSign = true,
            .thousandsSeparator = commonThouSep,
            .thousandsSeparatorFrac = commonThouSepFrac,
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
            .style = NumberStyle::normal,
            .precision = 1,
            .plusSign = false,
            .zeroMode = ZeroMode::alwaysPositive,
            .unicodeMinusSign = true,
            .thousandsSeparator = commonThouSep,
            .thousandsSeparatorFrac = commonThouSepFrac,
            .leadingZero = true,
            .stripTrailingZeroes = true,
            .degreesMode = {},
        };
    }
    else if constexpr ( std::is_same_v<E, MovementSpeedUnit> )
    {
        return UnitToStringParams<MovementSpeedUnit>{
            .sourceUnit = std::nullopt,
            .targetUnit = MovementSpeedUnit::mmPerSecond,
            .unitSuffix = true,
            .style = NumberStyle::normal,
            .precision = 3,
            .plusSign = false,
            .zeroMode = ZeroMode::alwaysPositive,
            .unicodeMinusSign = true,
            .thousandsSeparator = commonThouSep,
            .thousandsSeparatorFrac = commonThouSepFrac,
            .leadingZero = true,
            .stripTrailingZeroes = true,
            .degreesMode = {},
        };
    }
    else if constexpr ( std::is_same_v<E, AreaUnit> )
    {
        return UnitToStringParams<AreaUnit>{
            .sourceUnit = std::nullopt,
            .targetUnit = AreaUnit::mm2,
            .unitSuffix = true,
            .style = NumberStyle::normal,
            .precision = 3,
            .plusSign = false,
            .zeroMode = ZeroMode::alwaysPositive,
            .unicodeMinusSign = true,
            .thousandsSeparator = commonThouSep,
            .thousandsSeparatorFrac = commonThouSepFrac,
            .leadingZero = true,
            .stripTrailingZeroes = true,
            .degreesMode = {},
        };
    }
    else if constexpr ( std::is_same_v<E, VolumeUnit> )
    {
        return UnitToStringParams<VolumeUnit>{
            .sourceUnit = std::nullopt,
            .targetUnit = VolumeUnit::mm3,
            .unitSuffix = true,
            .style = NumberStyle::normal,
            .precision = 3,
            .plusSign = false,
            .zeroMode = ZeroMode::alwaysPositive,
            .unicodeMinusSign = true,
            .thousandsSeparator = commonThouSep,
            .thousandsSeparatorFrac = commonThouSepFrac,
            .leadingZero = true,
            .stripTrailingZeroes = true,
            .degreesMode = {},
        };
    }
    else if constexpr ( std::is_same_v<E, InvLengthUnit> )
    {
        return UnitToStringParams<InvLengthUnit>{
            .sourceUnit = std::nullopt,
            .targetUnit = InvLengthUnit::inv_mm,
            .unitSuffix = true,
            .style = NumberStyle::normal,
            .precision = 3,
            .plusSign = false,
            .zeroMode = ZeroMode::alwaysPositive,
            .unicodeMinusSign = true,
            .thousandsSeparator = commonThouSep,
            .thousandsSeparatorFrac = commonThouSepFrac,
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
        { .conversionFactor = 1, .prettyName = "Millimeters", .unitSuffix = " mm" },
        { .conversionFactor = 1000, .prettyName = "Meters", .unitSuffix = " m" },
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
        { .conversionFactor = 0.01f, .prettyName = "Percents", .unitSuffix = " %" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( RatioUnit::_count ) );
    return ret[int( ratio )];
}
template <>
const UnitInfo& getUnitInfo( TimeUnit time )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = "Seconds", .unitSuffix = " s" },
        { .conversionFactor = 0.001f, .prettyName = "Milliseconds", .unitSuffix = " ms" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( TimeUnit::_count ) );
    return ret[int( time )];
}
template <>
const UnitInfo& getUnitInfo( MovementSpeedUnit speed )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = "Millimeters per second", .unitSuffix = " mm/s" },
        { .conversionFactor = 1000, .prettyName = "Meters per second", .unitSuffix = " m/s" },
        { .conversionFactor = 25.4f, .prettyName = "Inches per second", .unitSuffix = " in/s" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( MovementSpeedUnit::_count ) );
    return ret[int( speed )];
}
template <>
const UnitInfo& getUnitInfo( AreaUnit area )
{
    static const UnitInfo ret[] = {
        // U+00B2 SUPERSCRIPT TWO
        { .conversionFactor = 1, .prettyName = "Millimeters\xc2\xb2", .unitSuffix = " mm\xc2\xb2" },
        { .conversionFactor = 1000*1000, .prettyName = "Meters\xc2\xb2", .unitSuffix = " m\xc2\xb2" },
        { .conversionFactor = 25.4f*25.4f, .prettyName = "Inches\xc2\xb2", .unitSuffix = " in\xc2\xb2" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( AreaUnit::_count ) );
    return ret[int( area )];
}
template <>
const UnitInfo& getUnitInfo( VolumeUnit volume )
{
    static const UnitInfo ret[] = {
        // U+00B3 SUPERSCRIPT THREE
        { .conversionFactor = 1, .prettyName = "Millimeters\xc2\xb3", .unitSuffix = " mm\xc2\xb3" },
        { .conversionFactor = 1000*1000*1000, .prettyName = "Meters\xc2\xb3", .unitSuffix = " m\xc2\xb3" },
        { .conversionFactor = 25.4f*25.4f*25.4f, .prettyName = "Inches\xc2\xb3", .unitSuffix = " in\xc2\xb3" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( VolumeUnit::_count ) );
    return ret[int( volume )];
}
template <>
const UnitInfo& getUnitInfo( InvLengthUnit length )
{
    static const UnitInfo ret[] = {
        // U+207B SUPERSCRIPT MINUS, U+00B9 SUPERSCRIPT ONE
        { .conversionFactor = 1, .prettyName = "Millimeters\u207B\u00B9", .unitSuffix = " mm\u207B\u00B9" },
        { .conversionFactor = 1/1000.f, .prettyName = "Meters\u207B\u00B9", .unitSuffix = " m\u207B\u00B9" },
        { .conversionFactor = 1/25.4f, .prettyName = "Inches\u207B\u00B9", .unitSuffix = " in\u207B\u00B9" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( InvLengthUnit::_count ) );
    return ret[int( length )];
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

std::string_view toString( DegreesMode mode )
{
    switch ( mode )
    {
    case DegreesMode::degrees:
        return "Degrees";
    case DegreesMode::degreesMinutes:
        return "Degrees, minutes";
    case DegreesMode::degreesMinutesSeconds:
        return "Degrees, minutes, seconds";
    case DegreesMode::_count:
        break; // Nothing.
    }

    assert( false && "Unknown `DegreesMode` value." );
    return "??";
}

template <UnitEnum E, typename T>
static std::string valueToStringImpl( T value, const UnitToStringParams<E>& params )
{
    // If `str` starts with ASCII minus minus, and `params.unicodeMinusSign` is set, replace it with a Unicode minus sign.
    // Also strips the minus from negative zeroes if `params.allowNegativeZero` is false.
    auto adjustMinusSign = [&]( std::string& str )
    {
        auto onlyZeroes = [&]{ return std::all_of( str.begin(), str.end(), []( char ch ){ return bool( std::isdigit( (unsigned char)ch ) ) <=/*implies*/ ( ch == '0' ); } ); };

        switch ( params.zeroMode )
        {
        case ZeroMode::asIs:
            // Nothing.
            break;
        case ZeroMode::alwaysPositive:
            // If the only digits in `str` are zeroes and we don't allow negative zeroes, remove the minus sign.
            if (  str.starts_with( '-' ) && onlyZeroes() )
                str.erase( str.begin() );
            break;
        case ZeroMode::alwaysNegative:
            // If the only digits in `str` are zeroes and we don't allow positive zeroes, force the minus sign.
            if ( !str.starts_with( '-' ) && onlyZeroes() )
                str.insert( str.begin(), '-' );
            break;
        }

        // If there's no minus sign, add the plus sign.
        if ( params.plusSign && !str.starts_with( '-' ) )
            str.insert( str.begin(), '+' );

        // Replace the plain `-` sign with the fancy Unicode one.
        if ( params.unicodeMinusSign && str.starts_with( '-' ) )
        {
            // U+2212 MINUS SIGN.
            str[0] = '\xe2';
            str.insert( str.begin() + 1, { '\x88', '\x92' } );
        }
    };

    // `str` is a value representing arcseconds or arcminutes.
    // 1. Pads it with zeroes on the left to have at least two digits before the point.
    // 2. Removes the leading minus, if any. (It should only be possible for the negative zero.)
    auto adjustArcMinutesOrSeconds = [&]( std::string& str )
    {
        if ( str.starts_with( '-' ) )
            str.erase( str.begin() );
        if ( std::isdigit( (unsigned char)str[0] ) && !std::isdigit( (unsigned char)str[1] ) )
            str = '0' + std::move( str );
    };

    std::string_view unitSuffix;
    if ( params.unitSuffix )
        unitSuffix = params.targetUnit ? getUnitInfo( *params.targetUnit ).unitSuffix : params.sourceUnit ? getUnitInfo( *params.sourceUnit ).unitSuffix : "";
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

                T wholeDegrees = 0;
                T minutes = std::modf( value, &wholeDegrees ) * 60;
                if ( minutes >= 59.5f )
                {
                    wholeDegrees += 1;
                    minutes = 0;
                }
                if ( negative )
                    wholeDegrees = -wholeDegrees;

                ret = fmt::format( "{:.0f}{}", wholeDegrees, getUnitInfo( AngleUnit::degrees ).unitSuffix );
                adjustMinusSign( ret );

                if ( params.degreesMode == DegreesMode::degreesMinutesSeconds )
                {
                    T wholeMinutes = 0;
                    T seconds = std::modf( minutes, &wholeMinutes ) * 60;
                    if ( seconds >= 59.5f )
                    {
                        wholeMinutes += 1;
                        seconds = 0;
                    }

                    std::string minutesStr = fmt::format( "{:.0f}'", wholeMinutes );
                    adjustArcMinutesOrSeconds( minutesStr );
                    ret += minutesStr;

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
            if constexpr ( std::is_same_v<E, AngleUnit> )
            {
                if ( params.degreesMode == DegreesMode::degreesMinutes || params.degreesMode == DegreesMode::degreesMinutesSeconds )
                {
                    std::string part = fmt::format( "{:.{}f}", value, precision );
                    adjustArcMinutesOrSeconds( part );
                    return part;
                }
            }

            if ( params.style == NumberStyle::maybeExponential )
                return fmt::format( "{:.{}g}", value, precision );
            else if ( params.style == NumberStyle::exponential )
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
    if ( params.style == NumberStyle::distributePrecision && fracPrecision > 0 )
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

        // Add the thousands separator.
        if ( params.thousandsSeparator || params.thousandsSeparatorFrac )
        {
            auto pos = formattedValue.find_first_of( ".eE" );

            if ( pos != std::string::npos )
            {
                // Add separator after the dot.

                if ( params.thousandsSeparatorFrac && formattedValue[pos] == '.' )
                {
                    while ( pos + 5 <= formattedValue.size() &&
                        std::all_of( formattedValue.begin() + pos + 1, formattedValue.begin() + pos + 5, []( unsigned char ch ){ return std::isdigit( ch ); } )
                    )
                    {
                        pos += 4;
                        formattedValue.insert( formattedValue.begin() + pos, params.thousandsSeparatorFrac );
                    }
                }
            }
            else
            {
                pos = formattedValue.size();
            }

            // Add separator before the dot.

            if ( params.thousandsSeparator )
            {
                while ( pos > 3 && std::isdigit( (unsigned char)formattedValue[pos - 4] ) )
                {
                    pos -= 3;
                    formattedValue.insert( formattedValue.begin() + pos, params.thousandsSeparator );
                }
            }
        }

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

        adjustMinusSign( formattedValue );

        ret += formattedValue;
    }

    ret += unitSuffix;

    if ( params.decorationFormatString != "{}" )
        return fmt::format( runtimeFmt( params.decorationFormatString ), ret );
    else
        return ret;
}

template <UnitEnum E, detail::Units::Scalar T>
std::string valueToString( T value, const UnitToStringParams<E>& params )
{
    // Convert to the target unit.
    if ( unitsAreEquivalent( params.sourceUnit, params.targetUnit ) )
    {
        // This can be integral or floating-point.
        return valueToStringImpl( value, params );
    }
    else
    {
        // This is always floating-point.
        return valueToStringImpl( convertUnits( params.sourceUnit, params.targetUnit, value ), params );
    }
}

#define MR_Y(T, E) template std::string valueToString<E, T>( T value, const UnitToStringParams<E>& params );
#define MR_X(E) DETAIL_MR_UNIT_VALUE_TYPES(MR_Y, E)
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X
#undef MR_Y

template <detail::Units::Scalar T>
std::string valueToString( T value, const VarUnitToStringParams& params )
{
    return std::visit( [&]( const auto& visitedParams )
    {
        return (valueToString)( value, visitedParams );
    }, params );
}

#define MR_X(T, unused) template std::string valueToString<T>( T value, const VarUnitToStringParams& params );
DETAIL_MR_UNIT_VALUE_TYPES(MR_X,)
#undef MR_X

template <detail::Units::Scalar T>
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
        if ( pos == std::string::npos ) // If too close to zero...
            return std::numeric_limits<T>::max_digits10;

        // pos - 2 == the number of leading zeroes in the fractional part, then +1 to show the first digit.
        return std::max( 0, int( pos ) - 1 );
    }
}

template <detail::Units::Scalar T>
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

#define MR_X(T, unused) \
    template int guessPrecision( T value ); \
    template int guessPrecision( T min, T max );
DETAIL_MR_UNIT_VALUE_TYPES(MR_X,)
#undef MR_X

template <UnitEnum E, detail::Units::Scalar T>
std::string valueToImGuiFormatString( T value, const UnitToStringParams<E>& params )
{
    std::string ret = replace( valueToString( value, params ), "%", "%%" );
    ret += "##%";

    if constexpr ( std::is_integral_v<T> )
    {
        using SignedT = std::make_signed_t<T>;
        if constexpr ( std::is_same_v<SignedT, signed char> )
            ret += "hh";
        else if constexpr ( std::is_same_v<SignedT, short> )
            ret += "h";
        else if constexpr ( std::is_same_v<SignedT, int> )
            ret += "";
        else if constexpr ( std::is_same_v<SignedT, long> )
            ret += "l";
        else if constexpr ( std::is_same_v<SignedT, long long> )
            ret += "ll";
        else
            static_assert( dependent_false<SignedT>, "Unknown integral type." );

        ret += std::is_signed_v<T> ? "d" : "u";
    }
    else
    {
        int precision = 0;
        std::size_t pos = ret.find( '.' );
        if ( pos != std::string::npos )
        {
            pos++;
            while (
                std::isdigit( (unsigned char)ret[pos + precision] ) ||
                ( params.thousandsSeparatorFrac && ret[pos + precision] == params.thousandsSeparatorFrac )
            )
                precision++;
        }

        fmt::format_to( std::back_inserter( ret ), ".{}", precision );

        if constexpr ( std::is_same_v<T, float> )
            ; // Nothing.
        else if constexpr ( std::is_same_v<T, double> )
            ; // Nothing. Same as for `float`, yes.
        else if constexpr ( std::is_same_v<T, long double> )
            ret += 'L';
        else
            static_assert( dependent_false<T>, "Unknown floating-point type." );

        if ( params.style == NumberStyle::exponential )
            ret += 'e';
        else if ( params.style == NumberStyle::maybeExponential )
            ret += 'g';
        else
            ret += 'f';
    }

    return ret;
}

#define MR_Y(T, E) template std::string valueToImGuiFormatString( T value, const UnitToStringParams<E>& params );
#define MR_X(E) DETAIL_MR_UNIT_VALUE_TYPES(MR_Y, E)
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X
#undef MR_Y

template <detail::Units::Scalar T>
std::string valueToImGuiFormatString( T value, const VarUnitToStringParams& params )
{
    return std::visit( [&]( const auto& visitedParams )
    {
        return (valueToImGuiFormatString)( value, visitedParams );
    }, params );
}

#define MR_X(T, unused) template std::string valueToImGuiFormatString<T>( T value, const VarUnitToStringParams& params );
DETAIL_MR_UNIT_VALUE_TYPES(MR_X,)
#undef MR_X

}
