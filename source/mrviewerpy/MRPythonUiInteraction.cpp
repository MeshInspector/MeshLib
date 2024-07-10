#include "MRMesh/MRPython.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRPythonAppendCommand.h"
#include "MRViewer/MRUITestEngine.h"
#include "MRViewer/MRViewer.h"
#include "MRPch/MRSpdlog.h"

#include <pybind11/stl.h>

#include <span>

namespace TestEngine = MR::UI::TestEngine;

namespace
{
    enum class EntryType
    {
        button,
        group,
        valueInt,
        valueUint,
        valueReal,
        valueString,
        other,
        // Don't forget to add new values to `pybind11::enum_` below!
    };

    struct TypedEntry
    {
        std::string name;
        EntryType type;
    };

    const TestEngine::GroupEntry& findGroup( std::span<const std::string> path )
    {
        const TestEngine::GroupEntry* cur = &TestEngine::getRootEntry();
        for ( const auto& segment : path )
            cur = &std::get<TestEngine::GroupEntry>( cur->elems.at( segment ).value );
        return *cur;
    }

    // Not using `MR_ADD_PYTHON_VEC` here, I don't seem to need any of custom functions it provides.
    std::vector<TypedEntry> listEntries( const std::vector<std::string>& path )
    {
        std::vector<TypedEntry> ret;
        MR::CommandLoop::runCommandFromGUIThread( [&]
        {
            const auto& group = findGroup( path );
            ret.reserve( group.elems.size() );
            for ( const auto& elem : group.elems )
            {
                ret.push_back( {
                    .name = elem.first,
                    .type = std::visit( MR::overloaded{
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
                        []( const auto& ) { return EntryType::other; },
                    }, elem.second.value ),
                } );
            }
        } );
        return ret;
    }

    void pressButton( const std::vector<std::string>& path )
    {
        if ( path.empty() )
            throw std::runtime_error( "Empty path not allowed here." );
        MR::CommandLoop::runCommandFromGUIThread( [&]
        {
            spdlog::info( "\n  Click: {}\n  Num Frame {}", path.back(), MR::getViewerInstance().getTotalFrames() );

            std::get<TestEngine::ButtonEntry>( findGroup( { path.data(), path.size() - 1 } ).elems.at( path.back() ).value ).simulateClick = true;
        } );
        for ( int i = 0; i < MR::getViewerInstance().forceRedrawMinimumIncrementAfterEvents; ++i )
            MR::CommandLoop::runCommandFromGUIThread( [] {} ); // wait frame
    }

    // Read/write values: (drags, sliders, etc)

    template <typename T>
    struct Value
    {
        T value = 0;
        T min = 0;
        T max = 0;
    };
    template <>
    struct Value<std::string>
    {
        std::string value;
    };
    using ValueInt = Value<std::int64_t>;
    using ValueUint = Value<std::uint64_t>;
    using ValueReal = Value<double>;
    using ValueString = Value<std::string>;

    template <typename T>
    Value<T> readValue( const std::vector<std::string>& path )
    {
        if ( path.empty() )
            throw std::runtime_error( "Empty path not allowed here." );
        Value<T> ret;
        MR::pythonAppendOrRun( [&]
        {
            const auto& entry = std::get<TestEngine::ValueEntry>( findGroup( { path.data(), path.size() - 1 } ).elems.at( path.back() ).value );

            // Try to read with the wrong signedness first.
            if constexpr ( std::is_same_v<T, std::int64_t> )
            {
                if ( auto val = std::get_if<TestEngine::ValueEntry::Value<std::uint64_t>>( &entry.value ) )
                {
                    // Allow if the value is not too large.
                    // We don't check if the max bound is too large, because it be too large by default if not specified.

                    if ( val->value > std::uint64_t( std::numeric_limits<std::int64_t>::max() ) )
                        throw std::runtime_error( "Attempt to read an uint64_t value as an int64_t, but the value is too large to fit into the target type. Read as uint64_t instead." );
                    ret.value = std::int64_t( val->value );
                    ret.min = std::int64_t( std::min( val->min, std::uint64_t( std::numeric_limits<std::int64_t>::max() ) ) );
                    ret.max = std::int64_t( std::min( val->max, std::uint64_t( std::numeric_limits<std::int64_t>::max() ) ) );
                    return;
                }
            }
            else if constexpr ( std::is_same_v<T, std::uint64_t> )
            {
                if ( auto val = std::get_if<TestEngine::ValueEntry::Value<std::int64_t>>( &entry.value ) )
                {
                    // Allow if the value is nonnegative, and the min bound is also nonnegative.

                    if ( val->value < 0 || val->min < 0 )
                        throw std::runtime_error( "Attempt to read an int64_t value as a uint64_t, but it is or can be negative. Read as int64_t instead." );
                    ret.value = std::uint64_t( val->value );
                    ret.min = std::uint64_t( val->min );
                    ret.max = std::uint64_t( val->max );
                    return;
                }
            }

            if ( auto val = std::get_if<TestEngine::ValueEntry::Value<T>>( &entry.value ) )
            {
                ret.value = val->value;
                if constexpr ( !std::is_same_v<T, std::string> )
                {
                    ret.min = val->min;
                    ret.max = val->max;
                }
                return;
            }

            throw std::runtime_error( std::is_floating_point_v<T>
                ? "Attempt to read an integer as a floating-point value."
                : "Attempt to read a floating-point value as an integer."
            );
        } );
        return ret;
    }

    template <typename T>
    void writeValue( const std::vector<std::string>& path, T value )
    {
        if ( path.empty() )
            throw std::runtime_error( "Empty path not allowed here." );
        MR::pythonAppendOrRun( [&]
        {
            const auto& entry = std::get<TestEngine::ValueEntry>( findGroup( { path.data(), path.size() - 1 } ).elems.at( path.back() ).value );

            bool usedDifferentSignedness = false;
            T simulatedValue{};
            T min{};
            T max{};

            // Try using the wrong signedness first.
            if constexpr ( std::is_same_v<T, std::int64_t> )
            {
                if ( auto val = std::get_if<TestEngine::ValueEntry::Value<std::uint64_t>>( &entry.value ) )
                {
                    // Allow if at least the min bound fits inside the input range.
                    if ( val->min > std::uint64_t( std::numeric_limits<std::int64_t>::max() ) )
                        throw std::runtime_error( "Attempt to write an int64_t into an uint64_t, but the min allowed value is larger than the max representable int64_t. Write as uint64_t instead." );

                    usedDifferentSignedness = true;
                    min = std::int64_t( val->min );
                    max = std::int64_t( std::min( val->max, std::uint64_t( std::numeric_limits<std::int64_t>::max() ) ) );
                }
            }
            else if constexpr ( std::is_same_v<T, std::uint64_t> )
            {
                if ( auto val = std::get_if<TestEngine::ValueEntry::Value<std::int64_t>>( &entry.value ) )
                {
                    // Allow if at least the max bound is nonnegative.
                    if ( val->max >= 0 )
                        throw std::runtime_error( "Attempt to write an uint64_t into an int64_t, but the max allowed value is negative. Write as uint64_t instead." );

                    usedDifferentSignedness = true;
                    min = std::uint64_t( std::max( val->min, std::int64_t( 0 ) ) );
                    max = std::uint64_t( val->max );
                }
            }

            // Use the exact type.
            if ( !usedDifferentSignedness )
            {
                auto opt = std::get_if<TestEngine::ValueEntry::Value<T>>( &entry.value );
                if ( !opt )
                {
                    throw std::runtime_error( std::is_floating_point_v<T>
                        ? "Attempt to write a floating-point value into an integer."
                        : "Attempt to write an integer into a floating-point value."
                    );
                }

                if constexpr ( !std::is_same_v<T, std::string> )
                {
                    min = opt->min;
                    max = opt->max;
                }
            }

            if constexpr ( !std::is_same_v<T, std::string> )
            {
                if ( value < min )
                    throw std::runtime_error( "The specified value is less than the min bound." );
                if ( value > max )
                    throw std::runtime_error( "The specified value is less than the max bound." );
            }
            simulatedValue = value;

            // Write the value back.
            if constexpr ( std::is_same_v<T, std::int64_t> )
            {
                if ( auto val = std::get_if<TestEngine::ValueEntry::Value<std::uint64_t>>( &entry.value ) )
                {
                    val->simulatedValue = std::uint64_t( simulatedValue );
                    return;
                }
            }
            else if constexpr ( std::is_same_v<T, std::uint64_t> )
            {
                if ( auto val = std::get_if<TestEngine::ValueEntry::Value<std::int64_t>>( &entry.value ) )
                {
                    val->simulatedValue = std::int64_t( simulatedValue );
                    return;
                }
            }

            std::get<TestEngine::ValueEntry::Value<T>>( entry.value ).simulatedValue = simulatedValue;
        } );
    }
}

MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiEntry, TypedEntry )
MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiValueInt, ValueInt )
MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiValueUint, ValueUint )
MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiValueReal, ValueReal )
MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiValueString, ValueString )

MR_ADD_PYTHON_CUSTOM_DEF( mrviewerpy, UiEntry, [] ( pybind11::module_& m )
{
    pybind11::enum_<EntryType>( m, "UiEntryType", "UI entry type enum." )
        .value( "button", EntryType::button )
        .value( "group", EntryType::group )
        .value( "valueInt", EntryType::valueInt )
        .value( "valueUint", EntryType::valueUint )
        .value( "valueReal", EntryType::valueReal )
        .value( "valueString", EntryType::valueString )
        .value( "other", EntryType::other )
    ;

    MR_PYTHON_CUSTOM_CLASS( UiValueInt ).def_readonly( "value", &ValueInt::value ).def_readonly( "min", &ValueInt::min ).def_readonly( "max", &ValueInt::max );
    MR_PYTHON_CUSTOM_CLASS( UiValueUint ).def_readonly( "value", &ValueUint::value ).def_readonly( "min", &ValueUint::min ).def_readonly( "max", &ValueUint::max );
    MR_PYTHON_CUSTOM_CLASS( UiValueReal ).def_readonly( "value", &ValueReal::value ).def_readonly( "min", &ValueReal::min ).def_readonly( "max", &ValueReal::max );
    MR_PYTHON_CUSTOM_CLASS( UiValueString ).def_readonly( "value", &ValueString::value );

    MR_PYTHON_CUSTOM_CLASS( UiEntry )
        .def_readonly( "name", &TypedEntry::name )
        .def_readonly( "type", &TypedEntry::type )
        .def("__repr__", []( const TypedEntry& e )
        {
            const char* typeString = nullptr;
            switch ( e.type )
            {
                case EntryType::button: typeString = "button"; break;
                case EntryType::valueInt: typeString = "valueInt"; break;
                case EntryType::valueUint: typeString = "valueUint"; break;
                case EntryType::valueReal: typeString = "valueReal"; break;
                case EntryType::valueString: typeString = "valueString"; break;
                case EntryType::group: typeString = "group"; break;
                case EntryType::other: typeString = "other"; break;
            }
            assert( typeString && "Unknown enum." );
            if ( !typeString )
                typeString = "??";

            return fmt::format( "<mrmesh.mrviewerpy.UiEntry '{}' of type '{}'>", e.name, typeString );
        } )
    ;
} )

MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiListEntries, listEntries,
    "List existing UI entries at the specified path.\n"
    "Pass an empty list to see top-level groups.\n"
    "Add group name to the end of the vector to see its contents.\n"
    "When you find the button you need, pass it to `uiPressButton()`."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiPressButton, pressButton,
    "Simulate a button click. Use `uiListEntries()` to find button names."
)

MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiReadValueInt, readValue<std::int64_t>,
    "Read a value from a drag/slider widget. This overload is for signed integers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiReadValueUint, readValue<std::uint64_t>,
    "Read a value from a drag/slider widget. This overload is for unsigned integers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiReadValueReal, readValue<double>,
    "Read a value from a drag/slider widget. This overload is for real numbers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiReadValueString, readValue<std::string>,
    "Read a value from a drag/slider widget. This overload is for strings."
)

MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiWriteValueInt, writeValue<std::int64_t>,
    "Write a value to a drag/slider widget. This overload is for signed integers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiWriteValueUint, writeValue<std::uint64_t>,
    "Write a value to a drag/slider widget. This overload is for unsigned integers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiWriteValueReal, writeValue<double>,
    "Write a value to a drag/slider widget. This overload is for real numbers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiWriteValueString, writeValue<std::string>,
    "Write a value to a drag/slider widget. This overload is for strings."
)
