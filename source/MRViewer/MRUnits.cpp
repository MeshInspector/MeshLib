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
            .fixedPrecision = false,
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
            .fixedPrecision = false,
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
            .fixedPrecision = true,
            .precision = 1,
            .unicodeMinusSign = true,
            .thousandsSeparator = 0,
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
        { .conversionFactor = 180, .prettyName = "Radians", .unitSuffix = " radians" },
        { .conversionFactor = PI_F, .prettyName = "Degrees", .unitSuffix = "\xC2\xB0" }, // U+00B0 DEGREE SIGN
    };
    static_assert( std::extent_v<decltype( ret )> == int( AngleUnit::_count ) );
    return ret[int( angle )];
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

template <UnitEnum E>
std::string valueToString( float value, const UnitToStringParams<E>& params )
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

    // Convert to the target unit.
    if ( params.sourceUnit )
        value = convertUnits( *params.sourceUnit, params.targetUnit, value );

    std::string_view unitSuffix;
    if ( params.unitSuffix )
        unitSuffix = getUnitInfo( params.targetUnit ).unitSuffix;
    std::string ret;

    // Handle arcseconds/arcminutes.
    // Write all but the last component to `ret. Set `value` and `unixSuffix` to the last component.
    if constexpr ( std::is_same_v<E, AngleUnit> )
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

    // Calculate precision after the decimal point.
    int fracPrecision = params.precision;
    if ( !params.fixedPrecision && fracPrecision > 0 )
    {
        int intDigits = 0;

        std::string tmp = fmt::format( "{:.{}f}", value, params.precision );
        std::size_t dot = tmp.find( '.' );
        if ( dot != std::string::npos )
            intDigits = int( dot ) - ( tmp.front() == '-' );

        fracPrecision -= intDigits;
    }
    if ( fracPrecision < 0 )
        fracPrecision = 0;

    { // Format the value.
        std::string formattedValue = fmt::format( "{:.{}f}", value, fracPrecision );

        // Remove the leading zero.
        if ( !params.leadingZero )
        {
            if ( formattedValue.starts_with( "0." ) )
                formattedValue.erase( formattedValue.begin() );
            else if ( formattedValue.starts_with( "-0." ) )
                formattedValue.erase( formattedValue.begin() + 1 );
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
        if ( params.stripTrailingZeroes && formattedValue.find( '.' ) != std::string::npos )
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

        adjustMinusSign( formattedValue );

        ret += formattedValue;
    }

    ret += unitSuffix;

    return ret;
}

#define MR_X(E) template std::string valueToString( float value, const UnitToStringParams<E>& params );
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X


}
