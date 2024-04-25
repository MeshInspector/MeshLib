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
}

MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiEntry, TypedEntry )
MR_ADD_PYTHON_CUSTOM_DEF( mrviewerpy, UiEntry, [] ( pybind11::module_& m )
{
    pybind11::enum_<EntryType>( m, "UiEntryType", "UI entry type enum." )
        .value( "button", EntryType::button )
        .value( "group", EntryType::group )
        .value( "other", EntryType::other )
    ;

    MR_PYTHON_CUSTOM_CLASS( UiEntry )
        .def_readonly( "name", &TypedEntry::name )
        .def_readonly( "type", &TypedEntry::type )
        .def("__repr__", []( const TypedEntry& e )
        {
            const char* typeString = nullptr;
            switch ( e.type )
            {
                case EntryType::button: typeString = "button"; break;
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
