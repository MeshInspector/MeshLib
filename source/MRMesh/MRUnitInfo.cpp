#include "MRUnitInfo.h"
#include "MRConstants.h"

namespace MR
{

// U+00B5 - Micro Sign, see https://www.utf8-chartable.de/
#define MR_MICRO_SIGN "\xC2\xB5"

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
        { .conversionFactor = 0.001f, .prettyName = _t( "Microns" ),     .unitSuffix = _t( " " MR_MICRO_SIGN "m" ) },
        { .conversionFactor = 1,      .prettyName = _t( "Millimeters" ), .unitSuffix = _t( " mm" ) },
        { .conversionFactor = 10,     .prettyName = _t( "Centimeters" ), .unitSuffix = _t( " cm" ) },
        { .conversionFactor = 1000,   .prettyName = _t( "Meters" ),      .unitSuffix = _t( " m" ) },
        { .conversionFactor = 25.4f,  .prettyName = _t( "Inches" ),      .unitSuffix = _t( " in" )/* or "\"" */ },
        { .conversionFactor = 304.8f, .prettyName = _t( "Feet" ),        .unitSuffix = _t( " ft" ) },
    };
    static_assert( std::extent_v<decltype( ret )> == int( LengthUnit::_count ) );
    return ret[int( length )];
}
template <>
const UnitInfo& getUnitInfo( AngleUnit angle )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = _t( "Radians" ), .unitSuffix = _t( " radians" ) },
        { .conversionFactor = PI_F/180.f, .prettyName = _t( "Degrees" ), .unitSuffix = _t( "\xC2\xB0" ) }, // U+00B0 DEGREE SIGN
    };
    static_assert( std::extent_v<decltype( ret )> == int( AngleUnit::_count ) );
    return ret[int( angle )];
}
template <>
const UnitInfo& getUnitInfo( PixelSizeUnit screenSize )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = _t( "Pixels" ), .unitSuffix = _t( " px" ) },
    };
    static_assert( std::extent_v<decltype( ret )> == int( PixelSizeUnit::_count ) );
    return ret[int( screenSize )];
}
template <>
const UnitInfo& getUnitInfo( RatioUnit ratio )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = _t( "Factor" ), .unitSuffix = _t( " x" ) },
        { .conversionFactor = 0.01f, .prettyName = _t( "Percents" ), .unitSuffix = _t( " %" ) },
    };
    static_assert( std::extent_v<decltype( ret )> == int( RatioUnit::_count ) );
    return ret[int( ratio )];
}
template <>
const UnitInfo& getUnitInfo( TimeUnit time )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 1, .prettyName = _t( "Seconds" ), .unitSuffix = _t( " s" ) },
        { .conversionFactor = 0.001f, .prettyName = _t( "Milliseconds" ), .unitSuffix = _t( " ms" ) },
    };
    static_assert( std::extent_v<decltype( ret )> == int( TimeUnit::_count ) );
    return ret[int( time )];
}
template <>
const UnitInfo& getUnitInfo( MovementSpeedUnit speed )
{
    static const UnitInfo ret[] = {
        { .conversionFactor = 0.001f, .prettyName = _t( "Microns per second" ),     .unitSuffix = _t( " " MR_MICRO_SIGN "m/s" ) },
        { .conversionFactor = 1,      .prettyName = _t( "Millimeters per second" ), .unitSuffix = _t( " mm/s" ) },
        { .conversionFactor = 10,     .prettyName = _t( "Centimeters per second" ), .unitSuffix = _t( " cm/s" ) },
        { .conversionFactor = 1000,   .prettyName = _t( "Meters per second" ),      .unitSuffix = _t( " m/s" ) },
        { .conversionFactor = 25.4f,  .prettyName = _t( "Inches per second" ),      .unitSuffix = _t( " in/s" ) },
        { .conversionFactor = 304.8f, .prettyName = _t( "Feet per second" ),        .unitSuffix = _t( " ft/s" ) },
    };
    static_assert( std::extent_v<decltype( ret )> == int( MovementSpeedUnit::_count ) );
    return ret[int( speed )];
}
template <>
const UnitInfo& getUnitInfo( AreaUnit area )
{
    static const UnitInfo ret[] = {
        // U+00B2 SUPERSCRIPT TWO
        { .conversionFactor = sqr(0.001f), .prettyName = _t( "Microns\xc2\xb2" ),     .unitSuffix = _t( " " MR_MICRO_SIGN "m\xc2\xb2" ) },
        { .conversionFactor = sqr(1.0f),   .prettyName = _t( "Millimeters\xc2\xb2" ), .unitSuffix = _t( " mm\xc2\xb2" ) },
        { .conversionFactor = sqr(10.0f),  .prettyName = _t( "Centimeters\xc2\xb2" ), .unitSuffix = _t( " cm\xc2\xb2" ) },
        { .conversionFactor = sqr(1000.0f),.prettyName = _t( "Meters\xc2\xb2" ),      .unitSuffix = _t( " m\xc2\xb2" ) },
        { .conversionFactor = sqr(25.4f),  .prettyName = _t( "Inches\xc2\xb2" ),      .unitSuffix = _t( " in\xc2\xb2" ) },
        { .conversionFactor = sqr(304.8f), .prettyName = _t( "Feet\xc2\xb2" ),        .unitSuffix = _t( " ft\xc2\xb2" ) },
    };
    static_assert( std::extent_v<decltype( ret )> == int( AreaUnit::_count ) );
    return ret[int( area )];
}
template <>
const UnitInfo& getUnitInfo( VolumeUnit volume )
{
    auto cbc = []( float x ) { return x * x * x; };
    static const UnitInfo ret[] = {
        // U+00B3 SUPERSCRIPT THREE
        { .conversionFactor = cbc(0.001f), .prettyName = _t( "Microns\xc2\xb3" ),     .unitSuffix = _t( " " MR_MICRO_SIGN "m\xc2\xb3" ) },
        { .conversionFactor = cbc(1.0f),   .prettyName = _t( "Millimeters\xc2\xb3" ), .unitSuffix = _t( " mm\xc2\xb3" ) },
        { .conversionFactor = cbc(10.0f),  .prettyName = _t( "Centimeters\xc2\xb3" ), .unitSuffix = _t( " cm\xc2\xb3" ) },
        { .conversionFactor = cbc(1000.0f),.prettyName = _t( "Meters\xc2\xb3" ),      .unitSuffix = _t( " m\xc2\xb3" ) },
        { .conversionFactor = cbc(25.4f),  .prettyName = _t( "Inches\xc2\xb3" ),      .unitSuffix = _t( " in\xc2\xb3" ) },
        { .conversionFactor = cbc(304.8f), .prettyName = _t( "Feet\xc2\xb3" ),        .unitSuffix = _t( " ft\xc2\xb3" ) },
    };
    static_assert( std::extent_v<decltype( ret )> == int( VolumeUnit::_count ) );
    return ret[int( volume )];
}
template <>
const UnitInfo& getUnitInfo( InvLengthUnit length )
{
    auto rep = []( float x ) { return 1 / x; };
    static const UnitInfo ret[] = {
        // U+207B SUPERSCRIPT MINUS, U+00B9 SUPERSCRIPT ONE
        { .conversionFactor = rep(0.001f), .prettyName = _t( "Microns\u207B\u00B9" ),     .unitSuffix = _t( " " MR_MICRO_SIGN "m\u207B\u00B9" ) },
        { .conversionFactor = rep(1.0f),   .prettyName = _t( "Millimeters\u207B\u00B9" ), .unitSuffix = _t( " mm\u207B\u00B9" ) },
        { .conversionFactor = rep(10.0f),  .prettyName = _t( "Centimeters\u207B\u00B9" ), .unitSuffix = _t( " cm\u207B\u00B9" ) },
        { .conversionFactor = rep(1000.0f),.prettyName = _t( "Meters\u207B\u00B9" ),      .unitSuffix = _t( " m\u207B\u00B9" ) },
        { .conversionFactor = rep(25.4f),  .prettyName = _t( "Inches\u207B\u00B9" ),      .unitSuffix = _t( " in\u207B\u00B9" ) },
        { .conversionFactor = rep(304.8f), .prettyName = _t( "Feet\u207B\u00B9" ),        .unitSuffix = _t( " ft\u207B\u00B9" ) },
    };
    static_assert( std::extent_v<decltype( ret )> == int( InvLengthUnit::_count ) );
    return ret[int( length )];
}

} // namespace MR
