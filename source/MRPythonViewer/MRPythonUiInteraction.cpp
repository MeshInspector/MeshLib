#include "MRPythonUiInteraction.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRUITestEngineControl.h"
#include "MRViewer/MRViewer.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRSpdlog.h"

#include <stdexcept>

namespace Control = MR::UI::TestEngine::Control;

namespace MR::PythonViewer
{

static_assert( int( UiEntryType::valueString ) == int( Control::EntryType::valueString ) );

namespace
{

// The module releases the GIL around the calls, so these can wait for the GUI thread.
template <typename F>
void runInGuiThread( F&& f )
{
    CommandLoop::runCommandFromGUIThread( std::forward<F>( f ) );
}

template <typename T>
T valueOrThrow( Expected<T> e )
{
    if ( !e )
        throw std::runtime_error( e.error() );
    return std::move( *e );
}

UiEntry toUiEntry( const Control::TypedEntry& e )
{
    return { .name = e.name, .type = UiEntryType( e.type ), .status = e.status };
}

template <typename T>
Control::Value<T> readValue( const std::vector<std::string>& path )
{
    Control::Value<T> ret;
    runInGuiThread( [&] { ret = valueOrThrow( Control::readValue<T>( path ) ); } );
    return ret;
}

template <typename T>
void writeValue( const std::vector<std::string>& path, T value )
{
    runInGuiThread( [&]
    {
        // Empty status = OK (write simulated); non-empty = disabled (silent no-op — pre-#5961 test contract).
        auto status = valueOrThrow( Control::writeValue<T>( path, std::move( value ) ) );
        if ( !status.empty() )
            spdlog::warn( "writeValue {}: {} (silent no-op)", Control::pathToString( path ), status );
    } );
}

} // namespace

std::vector<UiEntry> _uiListEntries( const std::vector<std::string>& path )
{
    std::vector<UiEntry> ret;
    runInGuiThread( [&]
    {
        for ( const auto& e : valueOrThrow( Control::listEntries( path ) ) )
            ret.push_back( toUiEntry( e ) );
    } );
    return ret;
}

std::vector<std::pair<std::vector<std::string>, UiEntry>> _uiListAllEntries( const std::vector<std::string>& rootPath )
{
    std::vector<std::pair<std::vector<std::string>, UiEntry>> ret;
    runInGuiThread( [&]
    {
        for ( const auto& [path, e] : valueOrThrow( Control::listAllEntries( rootPath ) ) )
            ret.emplace_back( path, toUiEntry( e ) );
    } );
    return ret;
}

void uiPressButton( const std::vector<std::string>& path )
{
    runInGuiThread( [&]
    {
        spdlog::info( "pressButton {}: frame {}", Control::pathToString( path ), getViewerInstance().getTotalFrames() );
        // Empty status = OK (click simulated); non-empty = disabled (silent no-op — pre-#5961 test contract).
        auto status = valueOrThrow( Control::pressButton( path ) );
        if ( !status.empty() )
            spdlog::warn( "pressButton {}: {} (silent no-op)", Control::pathToString( path ), status );
    } );
    for ( int i = 0; i < getViewerInstance().forceRedrawMinimumIncrementAfterEvents; ++i )
        runInGuiThread( [] {} ); // Wait a few frames.
}

UiValueInt uiReadValueInt( const std::vector<std::string>& path )
{
    auto v = readValue<std::int64_t>( path );
    return { v.value, v.min, v.max };
}

UiValueUint uiReadValueUint( const std::vector<std::string>& path )
{
    auto v = readValue<std::uint64_t>( path );
    return { v.value, v.min, v.max };
}

UiValueReal uiReadValueReal( const std::vector<std::string>& path )
{
    auto v = readValue<double>( path );
    return { v.value, v.min, v.max };
}

UiValueString uiReadValueString( const std::vector<std::string>& path )
{
    auto v = readValue<std::string>( path );
    return { std::move( v.value ), std::move( v.allowedValues ) };
}

void uiWriteValue( const std::vector<std::string>& path, std::int64_t value )
{
    writeValue( path, value );
}

void uiWriteValue( const std::vector<std::string>& path, std::uint64_t value )
{
    writeValue( path, value );
}

void uiWriteValue( const std::vector<std::string>& path, double value )
{
    writeValue( path, value );
}

void uiWriteValue( const std::vector<std::string>& path, const std::string& value )
{
    writeValue( path, value );
}

} // namespace MR::PythonViewer
