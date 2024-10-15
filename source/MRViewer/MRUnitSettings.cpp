#include "MRUnitSettings.h"

namespace MR::UnitSettings
{

static void forAllLengthUnits( auto&& func )
{
    // All length-related unit types must be listed here.
    func.template operator()<LengthUnit>();
    func.template operator()<AreaUnit>();
    func.template operator()<VolumeUnit>();
    func.template operator()<MovementSpeedUnit>();
    func.template operator()<InvLengthUnit>();
};
static void forAllAngleUnits( auto&& func )
{
    // All angle-related unit types must be listed here.
    func.template operator()<AngleUnit>();
};
static void forAllUnits( auto&& func )
{
    forAllLengthUnits( func );
    forAllAngleUnits( func );
    // All non-length/angle-related unit types must be listed here.
    func.template operator()<NoUnit>();
    func.template operator()<TimeUnit>();
    func.template operator()<RatioUnit>();
    func.template operator()<PixelSizeUnit>();
};

bool getShowLeadingZero()
{
    // Use the flag from an arbitrary unit, they should normally be the same on all units.
    return getDefaultUnitParams<LengthUnit>().leadingZero;
}

void setShowLeadingZero( bool show )
{
    forAllUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();
        params.leadingZero = show;
        setDefaultUnitParams( params );
    } );
}

char getThousandsSeparator( bool fractional )
{
    // Use the value from an arbitrary unit, they should normally be the same on all units.
    return fractional ? getDefaultUnitParams<LengthUnit>().thousandsSeparatorFrac : getDefaultUnitParams<LengthUnit>().thousandsSeparator;
}

void setThousandsSeparator( char ch, bool fractional )
{
    forAllUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();
        ( fractional ? params.thousandsSeparatorFrac : params.thousandsSeparator ) = ch;
        setDefaultUnitParams( params );
    } );
}

std::optional<LengthUnit> getUiLengthUnit()
{
    return getDefaultUnitParams<LengthUnit>().targetUnit;
}

void setUiLengthUnit( std::optional<LengthUnit> unit )
{
    // Override the leading zero. Everything except inches enables it.
    setShowLeadingZero( unit != LengthUnit::inches );

    auto getDependentUnit = overloaded{
        // All length-related unit types must be listed here.
        []<std::same_as<LengthUnit>>( LengthUnit unit )
        {
            return unit;
        },
        []<std::same_as<AreaUnit>>( LengthUnit unit )
        {
            switch ( unit )
            {
                case LengthUnit::mm:     return AreaUnit::mm2;
                case LengthUnit::meters: return AreaUnit::meters2;
                case LengthUnit::inches: return AreaUnit::inches2;
                case LengthUnit::_count: ; // MSVC warns otherwise.
            }
            assert( false );
            return AreaUnit::mm2;
        },
        []<std::same_as<VolumeUnit>>( LengthUnit unit )
        {
            switch ( unit )
            {
                case LengthUnit::mm:     return VolumeUnit::mm3;
                case LengthUnit::meters: return VolumeUnit::meters3;
                case LengthUnit::inches: return VolumeUnit::inches3;
                case LengthUnit::_count: ; // MSVC warns otherwise.
            }
            assert( false );
            return VolumeUnit::mm3;
        },
        []<std::same_as<MovementSpeedUnit>>( LengthUnit unit )
        {
            switch ( unit )
            {
                case LengthUnit::mm:     return MovementSpeedUnit::mmPerSecond;
                case LengthUnit::meters: return MovementSpeedUnit::metersPerSecond;
                case LengthUnit::inches: return MovementSpeedUnit::inchesPerSecond;
                case LengthUnit::_count: ; // MSVC warns otherwise.
            }
            assert( false );
            return MovementSpeedUnit::mmPerSecond;
        },
        []<std::same_as<InvLengthUnit>>( LengthUnit unit )
        {
            switch ( unit )
            {
                case LengthUnit::mm:     return InvLengthUnit::inv_mm;
                case LengthUnit::meters: return InvLengthUnit::inv_meters;
                case LengthUnit::inches: return InvLengthUnit::inv_inches;
                case LengthUnit::_count: ; // MSVC warns otherwise.
            }
            assert( false );
            return InvLengthUnit::inv_mm;
        },
    };

    forAllLengthUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();
        params.targetUnit = unit ? std::optional( getDependentUnit.template operator()<E>( *unit ) ) : std::nullopt;
        setDefaultUnitParams( params );
    } );
}

DegreesMode getDegreesMode()
{
    return getDefaultUnitParams<AngleUnit>().degreesMode;
}

void setDegreesMode( DegreesMode mode )
{
    forAllAngleUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();

        params.degreesMode = mode;

        if ( mode == DegreesMode::degrees )
            params.precision = 1;
        else
            params.precision = 0;

        params.style = NumberStyle::normal; // Just in case.

        setDefaultUnitParams( params );
    } );
}

int getUiLengthPrecision()
{
    // Use the value from an arbitrary length unit, they should normally be the same on all length units.
    return getDefaultUnitParams<LengthUnit>().precision;
}

void setUiLengthPrecision( int precision )
{
    forAllLengthUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();
        params.precision = precision;
        setDefaultUnitParams( params );
    } );
}

int getUiAnglePrecision()
{
    return getDefaultUnitParams<AngleUnit>().precision;
}

void setUiAnglePrecision( int precision )
{
    forAllAngleUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();
        params.precision = precision;
        setDefaultUnitParams( params );
    } );
}

}
