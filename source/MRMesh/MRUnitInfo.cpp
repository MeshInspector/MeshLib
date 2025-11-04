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
        { .conversionFactor = 0.001f, .prettyName = "Microns",     .unitSuffix = " " MR_MICRO_SIGN "m" },
        { .conversionFactor = 1,      .prettyName = "Millimeters", .unitSuffix = " mm" },
        { .conversionFactor = 10,     .prettyName = "Centimeters", .unitSuffix = " cm" },
        { .conversionFactor = 1000,   .prettyName = "Meters",      .unitSuffix = " m" },
        { .conversionFactor = 25.4f,  .prettyName = "Inches",      .unitSuffix = " in"/* or "\"" */ },
        { .conversionFactor = 304.8f, .prettyName = "Feet",        .unitSuffix = " ft" },
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
        { .conversionFactor = 0.001f, .prettyName = "Microns per second",     .unitSuffix = " " MR_MICRO_SIGN "m/s" },
        { .conversionFactor = 1,      .prettyName = "Millimeters per second", .unitSuffix = " mm/s" },
        { .conversionFactor = 10,     .prettyName = "Centimeters per second", .unitSuffix = " cm/s" },
        { .conversionFactor = 1000,   .prettyName = "Meters per second",      .unitSuffix = " m/s" },
        { .conversionFactor = 25.4f,  .prettyName = "Inches per second",      .unitSuffix = " in/s" },
        { .conversionFactor = 304.8f, .prettyName = "Feet per second",        .unitSuffix = " ft/s" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( MovementSpeedUnit::_count ) );
    return ret[int( speed )];
}
template <>
const UnitInfo& getUnitInfo( AreaUnit area )
{
    static const UnitInfo ret[] = {
        // U+00B2 SUPERSCRIPT TWO
        { .conversionFactor = sqr(0.001f), .prettyName = "Microns\xc2\xb2",     .unitSuffix = " " MR_MICRO_SIGN "m\xc2\xb2" },
        { .conversionFactor = sqr(1.0f),   .prettyName = "Millimeters\xc2\xb2", .unitSuffix = " mm\xc2\xb2" },
        { .conversionFactor = sqr(10.0f),  .prettyName = "Centimeters\xc2\xb2", .unitSuffix = " cm\xc2\xb2" },
        { .conversionFactor = sqr(1000.0f),.prettyName = "Meters\xc2\xb2",      .unitSuffix = " m\xc2\xb2" },
        { .conversionFactor = sqr(25.4f),  .prettyName = "Inches\xc2\xb2",      .unitSuffix = " in\xc2\xb2" },
        { .conversionFactor = sqr(304.8f), .prettyName = "Feet\xc2\xb2",        .unitSuffix = " ft\xc2\xb2" },
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
        { .conversionFactor = cbc(0.001f), .prettyName = "Microns\xc2\xb3",     .unitSuffix = " " MR_MICRO_SIGN "m\xc2\xb3" },
        { .conversionFactor = cbc(1.0f),   .prettyName = "Millimeters\xc2\xb3", .unitSuffix = " mm\xc2\xb3" },
        { .conversionFactor = cbc(10.0f),  .prettyName = "Centimeters\xc2\xb3", .unitSuffix = " cm\xc2\xb3" },
        { .conversionFactor = cbc(1000.0f),.prettyName = "Meters\xc2\xb3",      .unitSuffix = " m\xc2\xb3" },
        { .conversionFactor = cbc(25.4f),  .prettyName = "Inches\xc2\xb3",      .unitSuffix = " in\xc2\xb3" },
        { .conversionFactor = cbc(304.8f), .prettyName = "Feet\xc2\xb3",        .unitSuffix = " ft\xc2\xb3" },
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
        { .conversionFactor = rep(0.001f), .prettyName = "Microns\u207B\u00B9",     .unitSuffix = " " MR_MICRO_SIGN "m\u207B\u00B9" },
        { .conversionFactor = rep(1.0f),   .prettyName = "Millimeters\u207B\u00B9", .unitSuffix = " mm\u207B\u00B9" },
        { .conversionFactor = rep(10.0f),  .prettyName = "Centimeters\u207B\u00B9", .unitSuffix = " cm\u207B\u00B9" },
        { .conversionFactor = rep(1000.0f),.prettyName = "Meters\u207B\u00B9",      .unitSuffix = " m\u207B\u00B9" },
        { .conversionFactor = rep(25.4f),  .prettyName = "Inches\u207B\u00B9",      .unitSuffix = " in\u207B\u00B9" },
        { .conversionFactor = rep(304.8f), .prettyName = "Feet\u207B\u00B9",        .unitSuffix = " ft\u207B\u00B9" },
    };
    static_assert( std::extent_v<decltype( ret )> == int( InvLengthUnit::_count ) );
    return ret[int( length )];
}

} // namespace MR
