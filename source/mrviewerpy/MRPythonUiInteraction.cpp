#include "MRMesh/MRPython.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRPythonAppendCommand.h"
#include "MRViewer/MRUITestEngine.h"
#include "MRViewer/MRViewer.h"

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
        valueReal,
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
        MR::pythonAppendOrRun( [&]
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
                                []( const TestEngine::ValueEntry::Value<double>& ){ return EntryType::valueReal; },
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
        MR::pythonAppendOrRun( [&]
        {
            std::get<TestEngine::ButtonEntry>( findGroup( { path.data(), path.size() - 1 } ).elems.at( path.back() ).value ).simulateClick = true;
        } );
    }

    // Read/write values: (drags, sliders, etc)

    template <typename T>
    struct Value
    {
        T value = 0;
        T min = 0;
        T max = 0;
    };
    using ValueInt = Value<std::int64_t>;
    using ValueReal = Value<double>;

    template <typename T>
    Value<T> readValue( const std::vector<std::string>& path )
    {
        if ( path.empty() )
            throw std::runtime_error( "Empty path not allowed here." );
        Value<T> ret;
        MR::pythonAppendOrRun( [&]
        {
            const auto& entry = std::get<TestEngine::ValueEntry>( findGroup( { path.data(), path.size() - 1 } ).elems.at( path.back() ).value );
            const auto& val = std::get<TestEngine::ValueEntry::Value<T>>( entry.value );
            ret.value = val.value;
            ret.min = val.min;
            ret.max = val.max;
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
            const auto& val = std::get<TestEngine::ValueEntry::Value<T>>( entry.value );

            if ( value < val.min )
                throw std::runtime_error( "The specified value is less than the min bound." );
            if ( value > val.max )
                throw std::runtime_error( "The specified value is less than the max bound." );
            val.simulatedValue = value;
        } );
    }
}

MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiEntry, TypedEntry )
MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiValueInt, ValueInt )
MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiValueReal, ValueReal )

MR_ADD_PYTHON_CUSTOM_DEF( mrviewerpy, UiEntry, [] ( pybind11::module_& m )
{
    pybind11::enum_<EntryType>( m, "UiEntryType", "UI entry type enum." )
        .value( "button", EntryType::button )
        .value( "group", EntryType::group )
        .value( "valueInt", EntryType::valueInt )
        .value( "valueReal", EntryType::valueReal )
        .value( "other", EntryType::other )
    ;

    MR_PYTHON_CUSTOM_CLASS( UiValueInt ).def_readonly( "value", &ValueInt::value ).def_readonly( "min", &ValueInt::min ).def_readonly( "max", &ValueInt::max );
    MR_PYTHON_CUSTOM_CLASS( UiValueReal ).def_readonly( "value", &ValueReal::value ).def_readonly( "min", &ValueReal::min ).def_readonly( "max", &ValueReal::max );

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
                case EntryType::valueReal: typeString = "valueReal"; break;
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
    "Read a value from a drag/slider widget. This overload is for integers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiReadValueReal, readValue<double>,
    "Read a value from a drag/slider widget. This overload is for real numbers."
)

MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiWriteValueInt, writeValue<std::int64_t>,
    "Write a value to a drag/slider widget. This overload is for integers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiWriteValueReal, writeValue<double>,
    "Write a value to a drag/slider widget. This overload is for real numbers."
)
