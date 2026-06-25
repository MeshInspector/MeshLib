#include "MRUITestEngineControl.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRUITestEngine.h"

#include <algorithm>
#include <span>
#include <utility>
#include <variant>

namespace MR::UI::TestEngine::Control
{

static std::string listKeys( const MR::UI::TestEngine::GroupEntry& group )
{
    std::string ret;
    bool first = true;
    for ( const auto& elem : group.elems )
    {
        if ( first )
            first = false;
        else
            ret += ", ";
        ret += '`';
        ret += elem.first;
        ret += '`';
    }
    return ret;
}

static const Expected<const TestEngine::GroupEntry *> findGroup( std::span<const std::string> path )
{
    const TestEngine::GroupEntry* cur = &TestEngine::getRootEntry();
    for ( const auto& segment : path )
    {
        auto iter = cur->elems.find( segment );
        if ( iter == cur->elems.end() )
            return unexpected( fmt::format( "No such entry: `{}`. Known entries are: {}.", segment, listKeys( *cur ) ) );
        auto ex = iter->second.getAs<TestEngine::GroupEntry>( segment );
        if (!ex)
            return unexpected( ex.error() );
        cur = *ex;
    }
    return cur;
}

std::string pathToString( const std::vector<std::string>& path )
{
    std::string pathString;
    for ( const auto & s : path )
    {
        if ( !pathString.empty() )
            pathString += '/';
        pathString += s;
    }
    return pathString;
}

// Read the disabled-reason string off an internal entry. Empty means the entry accepts input.
// Groups are never disabled.
static std::string_view entryDisabledReason( const TestEngine::Entry& entry )
{
    return std::visit( MR::overloaded{
        []( const TestEngine::ButtonEntry& e ) -> std::string_view { return e.disabledReason; },
        []( const TestEngine::ValueEntry& e )  -> std::string_view { return e.disabledReason; },
        []( const TestEngine::GroupEntry& )    -> std::string_view { return {}; },
    }, entry.value );
}

// Compose the single user-facing status string from the entry's `disabledReason`.
// The reason covers every disable source — caller-supplied requirements, ImGui::BeginDisabled
// auto-detect, and "blocked by modal '<name>'" (all formatted at registration time in
// `MRUITestEngine.cpp:effectiveDisabledReason`).
static std::string composeStatus( std::string_view disabledReason )
{
    if ( !disabledReason.empty() )
        return fmt::format( "disabled: {}", disabledReason );
    return "available";
}

static EntryType typeOf( const TestEngine::Entry& entry )
{
    return std::visit( MR::overloaded{
        []( const TestEngine::ButtonEntry& ) { return EntryType::button; },
        []( const TestEngine::ValueEntry& e )
        {
            return std::visit( MR::overloaded{
                []( const TestEngine::ValueEntry::Value<std::int64_t>& ){ return EntryType::valueInt; },
                []( const TestEngine::ValueEntry::Value<std::uint64_t>& ){ return EntryType::valueUint; },
                []( const TestEngine::ValueEntry::Value<double>& ){ return EntryType::valueReal; },
                []( const TestEngine::ValueEntry::Value<std::string>& ){ return EntryType::valueString; },
            }, e.value );
        },
        []( const TestEngine::GroupEntry& ) { return EntryType::group; },
    }, entry.value );
}

Expected<std::vector<TypedEntry>> listEntries( const std::vector<std::string>& path )
{
    auto groupEx = findGroup( path );
    if ( !groupEx )
        return unexpected( groupEx.error() );
    const auto& group = **groupEx;

    std::vector<TypedEntry> ret;
    ret.reserve( group.elems.size() );

    for ( const auto& elem : group.elems )
    {
        ret.push_back( {
            .name = elem.first,
            .type = typeOf( elem.second ),
            .status = composeStatus( entryDisabledReason( elem.second ) ),
        } );
    }
    return ret;
}

static void walkAll( std::vector<std::string>& pathStack,
                     const TestEngine::Entry& entry,
                     std::vector<PathedEntry>& out )
{
    TypedEntry te{
        .name   = pathStack.back(),
        .type   = typeOf( entry ),
        .status = composeStatus( entryDisabledReason( entry ) ),
    };
    out.emplace_back( pathStack, std::move( te ) );

    if ( auto g = std::get_if<TestEngine::GroupEntry>( &entry.value ) )
    {
        for ( const auto& [childName, childEntry] : g->elems )
        {
            pathStack.push_back( childName );
            walkAll( pathStack, childEntry, out );
            pathStack.pop_back();
        }
    }
}

Expected<std::vector<PathedEntry>> listAllEntries( const std::vector<std::string>& rootPath )
{
    auto groupEx = findGroup( rootPath );
    if ( !groupEx )
        return unexpected( groupEx.error() );
    const auto& group = **groupEx;

    std::vector<PathedEntry> ret;
    std::vector<std::string> pathStack = rootPath;
    for ( const auto& [childName, childEntry] : group.elems )
    {
        pathStack.push_back( childName );
        walkAll( pathStack, childEntry, ret );
        pathStack.pop_back();
    }
    return ret;
}

Expected<std::string> pressButton( const std::vector<std::string>& path )
{
    if ( path.empty() )
        return unexpected( "pressButton: Empty path not allowed here." );

    auto groupEx = findGroup( { path.data(), path.size() - 1 } );
    if ( !groupEx )
        return unexpected( groupEx.error() );
    const auto& group = **groupEx;

    auto iter = group.elems.find( path.back() );
    if ( iter == group.elems.end() )
        return unexpected( fmt::format( "pressButton {}: no such entry: `{}`. Known entries are: {}.", pathToString( path ), path.back(), listKeys( group ) ) );

    auto buttonEx = iter->second.getAs<TestEngine::ButtonEntry>( path.back() );
    if ( !buttonEx )
        return unexpected( buttonEx.error() );

    const std::string_view disabledReason = ( *buttonEx )->disabledReason;
    if ( !disabledReason.empty() )
        return composeStatus( disabledReason );

    ( *buttonEx )->simulateClick = true;

    return {};
}

template <typename T>
Expected<Value<T>> readValue( const std::vector<std::string>& path )
{
    if ( path.empty() )
        return unexpected( "readValue: Empty path not allowed here." );

    auto groupEx = findGroup( { path.data(), path.size() - 1 } );
    if ( !groupEx )
        return unexpected( groupEx.error() );
    const auto& group = **groupEx;

    auto iter = group.elems.find( path.back() );
    if ( iter == group.elems.end() )
        return unexpected( fmt::format( "No such entry: `{}`. Known entries are: {}.", path.back(), listKeys( group ) ) );

    auto entryEx = iter->second.getAs<TestEngine::ValueEntry>( path.back() );
    if ( !entryEx )
        return unexpected( entryEx.error() );
    const auto& entry = **entryEx;

    if constexpr ( std::is_same_v<T, std::string> )
    {
        if ( auto val = std::get_if<TestEngine::ValueEntry::Value<T>>( &entry.value ) )
        {
            Value<T> ret;
            ret.value = val->value;
            ret.allowedValues = val->allowedValues;
            return ret;
        }

        return unexpected( "This isn't a string." );
    }
    else
    {
        // Try to read with the wrong signedness first.
        if constexpr ( std::is_same_v<T, std::int64_t> )
        {
            if ( auto val = std::get_if<TestEngine::ValueEntry::Value<std::uint64_t>>( &entry.value ) )
            {
                // Allow if the value is not too large.
                // We don't check if the max bound is too large, because it be too large by default if not specified.

                if ( val->value > std::uint64_t( std::numeric_limits<std::int64_t>::max() ) )
                    return unexpected( "Attempt to read an uint64_t value as an int64_t, but the value is too large to fit into the target type. Read as uint64_t instead." );

                Value<T> ret;
                ret.value = std::int64_t( val->value );
                ret.min = std::int64_t( std::min( val->min, std::uint64_t( std::numeric_limits<std::int64_t>::max() ) ) );
                ret.max = std::int64_t( std::min( val->max, std::uint64_t( std::numeric_limits<std::int64_t>::max() ) ) );
                return ret;
            }
        }
        else if constexpr ( std::is_same_v<T, std::uint64_t> )
        {
            if ( auto val = std::get_if<TestEngine::ValueEntry::Value<std::int64_t>>( &entry.value ) )
            {
                // Allow if the value is nonnegative, and the min bound is also nonnegative.

                if ( val->value < 0 || val->min < 0 )
                    return unexpected( "Attempt to read an int64_t value as a uint64_t, but it is or can be negative. Read as int64_t instead." );

                Value<T> ret;
                ret.value = std::uint64_t( val->value );
                ret.min = std::uint64_t( val->min );
                ret.max = std::uint64_t( val->max );
                return ret;
            }
        }

        if ( auto val = std::get_if<TestEngine::ValueEntry::Value<T>>( &entry.value ) )
        {
            Value<T> ret;
            ret.value = val->value;
            ret.min = val->min;
            ret.max = val->max;
            return ret;
        }

        return unexpected( std::is_floating_point_v<T>
            ? "This isn't a floating-point value."
            : "This isn't an integer."
        );
    }
}

template Expected<Value<std::int64_t >> readValue( const std::vector<std::string>& path );
template Expected<Value<std::uint64_t>> readValue( const std::vector<std::string>& path );
template Expected<Value<double       >> readValue( const std::vector<std::string>& path );
template Expected<Value<std::string  >> readValue( const std::vector<std::string>& path );


template <typename T>
Expected<std::string> writeValue( const std::vector<std::string>& path, T value )
{
    if ( path.empty() )
        return unexpected( "writeValue: Empty path not allowed here." );

    auto groupEx = findGroup( { path.data(), path.size() - 1 } );
    if ( !groupEx )
        return unexpected( groupEx.error() );
    const auto& group = **groupEx;

    auto iter = group.elems.find( path.back() );
    if ( iter == group.elems.end() )
        return unexpected( fmt::format( "writeValue {}: no such entry: `{}`. Known entries are: {}.", pathToString( path ), path.back(), listKeys( group ) ) );

    auto entryEx = iter->second.getAs<TestEngine::ValueEntry>( path.back() );
    if ( !entryEx )
        return unexpected( entryEx.error() );
    const auto& entry = **entryEx;

    const std::string_view disabledReason = entry.disabledReason;
    if ( !disabledReason.empty() )
        return composeStatus( disabledReason );

    auto writeValueOfCorrectType = [&entry, &path]( auto fixedValue ) -> Expected<std::string>
    {
        using U = decltype( fixedValue );
        auto &target = std::get<TestEngine::ValueEntry::Value<U>>( entry.value );

        // Validate the value.
        if constexpr ( std::is_same_v<U, std::string> )
        {
            if ( target.allowedValues && std::find( target.allowedValues->begin(), target.allowedValues->end(), fixedValue ) == target.allowedValues->end() )
            {
                std::string allowedValuesStr;
                bool first = true;
                for ( const auto& allowedValue : *target.allowedValues )
                {
                    if ( !std::exchange( first, false ) )
                        allowedValuesStr += ", ";

                    allowedValuesStr += '`';
                    allowedValuesStr += allowedValue;
                    allowedValuesStr += '`';
                }

                return unexpected( fmt::format( "writeValue {}: string `{}` is not allowed here. Allowed values: {}.", pathToString( path ), fixedValue, allowedValuesStr ) );
            }
        }
        else
        {
            if ( fixedValue < target.min )
                return unexpected( fmt::format( "writeValue {}: the specified value {} is less than the min bound {}.", pathToString( path ), fixedValue, target.min ) );
            if ( fixedValue > target.max )
                return unexpected( fmt::format( "writeValue {}: the specified value {} is more than the max bound {}.", pathToString( path ), fixedValue, target.max ) );
        }

        std::get<TestEngine::ValueEntry::Value<U>>( entry.value ).simulatedValue = std::move( fixedValue );

        return {};
    };

    if constexpr ( std::is_same_v<T, std::string> )
    {
        if ( std::holds_alternative<TestEngine::ValueEntry::Value<std::string>>( entry.value ) )
            return writeValueOfCorrectType( std::move( value ) );
        else
            return unexpected( fmt::format( "writeValue: `{}` is a number, but received a string.", pathToString( path ) ) );
    }
    else if constexpr ( std::is_same_v<T, double> )
    {
        return std::visit( MR::overloaded{
            [&]( const TestEngine::ValueEntry::Value<std::string  >& ) -> Expected<std::string> { return unexpected( fmt::format( "writeValue: `{}` is a string, but received a number.", pathToString( path ) ) ); },
            [&]( const TestEngine::ValueEntry::Value<double       >& ) -> Expected<std::string> { return writeValueOfCorrectType( value ); },
            [&]( const TestEngine::ValueEntry::Value<std::int64_t >& ) -> Expected<std::string> { return unexpected( fmt::format( "writeValue: `{}` is an integer, but received a fractional number.", pathToString( path ) ) ); },
            [&]( const TestEngine::ValueEntry::Value<std::uint64_t>& ) -> Expected<std::string> { return unexpected( fmt::format( "writeValue: `{}` is an integer, but received a fractional number.", pathToString( path ) ) ); },
        }, entry.value );
    }
    else if constexpr ( std::is_same_v<T, std::int64_t> )
    {
        return std::visit( MR::overloaded{
            [&]( const TestEngine::ValueEntry::Value<std::string  >& ) -> Expected<std::string> { return unexpected( fmt::format( "writeValue: `{}` is a string, but received a number.", pathToString( path ) ) ); },
            [&]( const TestEngine::ValueEntry::Value<double       >& ) -> Expected<std::string> { return writeValueOfCorrectType( double( value ) ); },
            [&]( const TestEngine::ValueEntry::Value<std::int64_t >& ) -> Expected<std::string> { return writeValueOfCorrectType( value ); },
            [&]( const TestEngine::ValueEntry::Value<std::uint64_t>& ) -> Expected<std::string>
            {
                if ( value < 0 )
                    return unexpected( fmt::format( "writeValue: `{}` is unsigned, but received a negative number.", pathToString( path ) ) );
                return writeValueOfCorrectType( std::uint64_t( value ) );
            },
        }, entry.value );
    }
    else if constexpr ( std::is_same_v<T, std::uint64_t> )
    {
        return std::visit( MR::overloaded{
            [&]( const TestEngine::ValueEntry::Value<std::string  >& ) -> Expected<std::string> { return unexpected( fmt::format( "writeValue: `{}` is a string, but received a number.", pathToString( path ) ) ); },
            [&]( const TestEngine::ValueEntry::Value<double       >& ) -> Expected<std::string> { return writeValueOfCorrectType( double( value ) ); },
            [&]( const TestEngine::ValueEntry::Value<std::uint64_t>& ) -> Expected<std::string> { return writeValueOfCorrectType( value ); },
            [&]( const TestEngine::ValueEntry::Value<std::int64_t >& ) -> Expected<std::string>
            {
                if ( value > std::uint64_t( std::numeric_limits<std::int64_t>::max() ) )
                    return unexpected( fmt::format( "writeValue: `{}` is signed, but received an unsigned integer large enough to not be representable as `int64_t`.", pathToString( path ) ) );
                return writeValueOfCorrectType( std::int64_t( value ) );
            },
        }, entry.value );
    }
}

template Expected<std::string> writeValue( const std::vector<std::string>& path, std::int64_t  value );
template Expected<std::string> writeValue( const std::vector<std::string>& path, std::uint64_t value );
template Expected<std::string> writeValue( const std::vector<std::string>& path, double        value );
template Expected<std::string> writeValue( const std::vector<std::string>& path, std::string   value );

} // namespace MR::UI::TestEngine::Control
