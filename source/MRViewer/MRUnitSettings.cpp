#include "MRUnitSettings.h"

#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewerSettingsManager.h"

namespace MR::UnitSettings
{

static const std::string
    cSettingLeadingZero = "units.leadingZero",
    cSettingThouSep = "units.thousandsSeparator",
    cSettingUnitLen = "units.unitLength",
    cSettingDegreesMode = "units.degreesMode",
    cSettingPrecisionLen = "units.precisionLength",
    cSettingPrecisionAngle = "units.precisionAngle";

static void forAllLengthUnits( auto&& func )
{
    // All length-related unit types must be listed here.
    func.template operator()<LengthUnit>();
    func.template operator()<AreaUnit>();
    func.template operator()<VolumeUnit>();
    func.template operator()<MovementSpeedUnit>();
    func.template operator()<InvLengthUnit>();
}
static void forAllAngleUnits( auto&& func )
{
    // All angle-related unit types must be listed here.
    func.template operator()<AngleUnit>();
}
static void forAllUnits( auto&& func )
{
    forAllLengthUnits( func );
    forAllAngleUnits( func );
    // All non-length/angle-related unit types must be listed here.
    func.template operator()<NoUnit>();
    func.template operator()<TimeUnit>();
    func.template operator()<RatioUnit>();
    func.template operator()<PixelSizeUnit>();
}

void loadFromViewerSettings()
{
    auto& m = getViewerInstance().getViewerSettingsManager();
    if ( !m )
        return;

    setShowLeadingZero( m->loadBool( cSettingLeadingZero, true ), WriteToSettings::no );

    // The order here can be important, because setting the length automatically sets the preferred leading zero,
    // and setting the degrees mode automatically sets the preferred angle precision.

    { // Length unit.
        static const std::unordered_map<std::string, LengthUnit> map = []{
            std::unordered_map<std::string, LengthUnit> ret;
            for ( int i = 0; i < int( LengthUnit::_count ); i++ )
                ret.try_emplace( std::string( getUnitInfo( LengthUnit( i ) ).prettyName ), LengthUnit( i ) );
            return ret;
        }();
        auto it = map.find( m->loadString( cSettingUnitLen, "" ) );
        setUiLengthUnit( it != map.end() ? it->second : LengthUnit::mm, true, WriteToSettings::no );
    }

    { // Thousands separator.
        std::string str = m->loadString( cSettingThouSep, " " );
        if ( str.empty() )
            setThousandsSeparator( 0, WriteToSettings::no );
        else if ( str.size() == 1 )
            setThousandsSeparator( str.front(), WriteToSettings::no );
    }

    { // Degrees mode.
        static const std::unordered_map<std::string, DegreesMode> map = []{
            std::unordered_map<std::string, DegreesMode> ret;
            for ( int i = 0; i < int( DegreesMode::_count ); i++ )
                ret.try_emplace( std::string( toString( DegreesMode( i ) ) ), DegreesMode( i ) );
            return ret;
        }();
        auto it = map.find( m->loadString( cSettingDegreesMode, "" ) );
        setDegreesMode( it != map.end() ? it->second : DegreesMode::degrees, true, WriteToSettings::no );
    }

    // Precision.
    if ( int p = m->loadInt( cSettingPrecisionLen, -1 ); p >= 0 )
        setUiLengthPrecision( p, WriteToSettings::no );
    if ( int p = m->loadInt( cSettingPrecisionAngle, -1 ); p >= 0 )
        setUiAnglePrecision( p, WriteToSettings::no );
}

bool getShowLeadingZero()
{
    // Use the flag from an arbitrary unit, they should normally be the same on all units.
    return getDefaultUnitParams<LengthUnit>().leadingZero;
}

void setShowLeadingZero( bool show, WriteToSettings writeToSettings )
{
    forAllUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();
        params.leadingZero = show;
        setDefaultUnitParams( params );
    } );

    if ( bool( writeToSettings ) )
    {
        if ( auto &m = getViewerInstance().getViewerSettingsManager() )
            m->saveBool( cSettingLeadingZero, show );
    }
}

char getThousandsSeparator()
{
    // Use the value from an arbitrary unit, they should normally be the same on all units.
    return getDefaultUnitParams<LengthUnit>().thousandsSeparator;
}

void setThousandsSeparator( char ch, WriteToSettings writeToSettings )
{
    forAllUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();
        params.thousandsSeparator = ch;
        setDefaultUnitParams( params );
    } );

    if ( bool( writeToSettings ) )
    {
        if ( auto &m = getViewerInstance().getViewerSettingsManager() )
            m->saveString( cSettingThouSep, std::string( 1, ch ) );
    }
}

std::optional<LengthUnit> getUiLengthUnit()
{
    return getDefaultUnitParams<LengthUnit>().targetUnit;
}

void setUiLengthUnit( std::optional<LengthUnit> unit, bool setPreferredLeadingZero, WriteToSettings writeToSettings )
{
    // Override the leading zero. Everything except inches enables it.
    if ( setPreferredLeadingZero )
        setShowLeadingZero( unit != LengthUnit::inches, writeToSettings );

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

    if ( bool( writeToSettings ) )
    {
        if ( auto &m = getViewerInstance().getViewerSettingsManager() )
            m->saveString( cSettingUnitLen, unit ? std::string( getUnitInfo( *unit ).prettyName ) : "none" );
    }
}

DegreesMode getDegreesMode()
{
    return getDefaultUnitParams<AngleUnit>().degreesMode;
}

void setDegreesMode( DegreesMode mode, bool setPreferredPrecision, WriteToSettings writeToSettings )
{
    forAllAngleUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();

        params.degreesMode = mode;

        if ( setPreferredPrecision )
        {
            if ( mode == DegreesMode::degrees )
                params.precision = 1;
            else
                params.precision = 0;
        }

        params.style = NumberStyle::normal; // Just in case.

        setDefaultUnitParams( params );
    } );

    if ( bool( writeToSettings ) )
    {
        if ( auto &m = getViewerInstance().getViewerSettingsManager() )
            m->saveString( cSettingDegreesMode, std::string( toString( mode ) ) );
    }
}

int getUiLengthPrecision()
{
    // Use the value from an arbitrary length unit, they should normally be the same on all length units.
    return getDefaultUnitParams<LengthUnit>().precision;
}

void setUiLengthPrecision( int precision, WriteToSettings writeToSettings )
{
    forAllLengthUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();
        params.precision = precision;
        setDefaultUnitParams( params );
    } );

    if ( bool( writeToSettings ) )
    {
        if ( auto &m = getViewerInstance().getViewerSettingsManager() )
            m->saveInt( cSettingPrecisionLen, precision );
    }
}

int getUiAnglePrecision()
{
    return getDefaultUnitParams<AngleUnit>().precision;
}

void setUiAnglePrecision( int precision, WriteToSettings writeToSettings )
{
    forAllAngleUnits( [&]<typename E>()
    {
        auto params = getDefaultUnitParams<E>();
        params.precision = precision;
        setDefaultUnitParams( params );
    } );

    if ( bool( writeToSettings ) )
    {
        if ( auto &m = getViewerInstance().getViewerSettingsManager() )
            m->saveInt( cSettingPrecisionAngle, precision );
    }
}

}
