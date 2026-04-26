#include "MRPython/MRPython.h"
#include "MRViewer/MRPythonAppendCommand.h"
#include "MRViewer/MRUITestEngineControl.h"
#include "MRViewer/MRViewer.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRSpdlog.h"

#include <pybind11/stl.h>

#include <span>

namespace Control = MR::UI::TestEngine::Control;

MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiEntry, Control::TypedEntry )
MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiValueInt, Control::ValueInt )
MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiValueUint, Control::ValueUint )
MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiValueReal, Control::ValueReal )
MR_ADD_PYTHON_CUSTOM_CLASS( mrviewerpy, UiValueString, Control::ValueString )

MR_ADD_PYTHON_CUSTOM_DEF( mrviewerpy, UiEntry, [] ( pybind11::module_& m )
{
    // Not using `MR_ADD_PYTHON_VEC(..., TypedEntry)` here, I don't seem to need any of custom functions it provides.

    pybind11::enum_<Control::EntryType>( m, "UiEntryType", "UI entry type enum." )
        .value( "button", Control::EntryType::button )
        .value( "group", Control::EntryType::group )
        .value( "valueInt", Control::EntryType::valueInt )
        .value( "valueUint", Control::EntryType::valueUint )
        .value( "valueReal", Control::EntryType::valueReal )
        .value( "valueString", Control::EntryType::valueString )
    ;

    MR_PYTHON_CUSTOM_CLASS( UiValueInt ).def_readonly( "value", &Control::ValueInt::value ).def_readonly( "min", &Control::ValueInt::min ).def_readonly( "max", &Control::ValueInt::max );
    MR_PYTHON_CUSTOM_CLASS( UiValueUint ).def_readonly( "value", &Control::ValueUint::value ).def_readonly( "min", &Control::ValueUint::min ).def_readonly( "max", &Control::ValueUint::max );
    MR_PYTHON_CUSTOM_CLASS( UiValueReal ).def_readonly( "value", &Control::ValueReal::value ).def_readonly( "min", &Control::ValueReal::min ).def_readonly( "max", &Control::ValueReal::max );
    MR_PYTHON_CUSTOM_CLASS( UiValueString ).def_readonly( "value", &Control::ValueString::value ).def_readonly( "allowed", &Control::ValueString::allowedValues );

    MR_PYTHON_CUSTOM_CLASS( UiEntry )
        .def_readonly( "name", &Control::TypedEntry::name )
        .def_readonly( "type", &Control::TypedEntry::type )
        .def_readonly( "status", &Control::TypedEntry::status )
        .def("__repr__", []( const Control::TypedEntry& e )
        {
            return fmt::format( "<mrmesh.mrviewerpy.UiEntry '{}' of type '{}' status='{}'>", e.name, toString( e.type ), e.status );
        } )
    ;
} )

MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiListEntries,
    []( const std::vector<std::string>& path )
    {
        std::vector<MR::UI::TestEngine::Control::TypedEntry> ret;
        MR::CommandLoop::runCommandFromGUIThread( [&]{ ret = MR::expectedValueOrThrow( MR::UI::TestEngine::Control::listEntries( path ) ); } );
        return ret;
    },
    "List existing UI entries at the specified path.\n"
    "Pass an empty list to see top-level groups.\n"
    "Add group name to the end of the vector to see its contents.\n"
    "When you find the button you need, pass it to `uiPressButton()`."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiListAllEntries,
    []( const std::vector<std::string>& rootPath )
    {
        std::vector<Control::PathedEntry> ret;
        MR::CommandLoop::runCommandFromGUIThread( [&]{ ret = MR::expectedValueOrThrow( Control::listAllEntries( rootPath ) ); } );
        return ret;
    },
    "Flat depth-first list of every UI entry in the subtree rooted at `rootPath`.\n"
    "Pass an empty list for the whole tree.\n"
    "Each element is a `(path, UiEntry)` tuple where `path[-1] == entry.name`."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiPressButton,
    []( const std::vector<std::string>& path )
    {
        MR::CommandLoop::runCommandFromGUIThread( [&]
        {
            spdlog::info( "pressButton {}: frame {}", MR::UI::TestEngine::Control::pathToString( path ), MR::getViewerInstance().getTotalFrames() );
            // Empty status = OK (click simulated); non-empty = disabled (silent no-op — pre-#5961 test contract).
            auto status = MR::expectedValueOrThrow( MR::UI::TestEngine::Control::pressButton( path ) );
            if ( !status.empty() )
                spdlog::warn( "pressButton {}: {} (silent no-op)", MR::UI::TestEngine::Control::pathToString( path ), status );
        } );
        for ( int i = 0; i < MR::getViewerInstance().forceRedrawMinimumIncrementAfterEvents; ++i )
            MR::CommandLoop::runCommandFromGUIThread( [] {} ); // Wait a few frames.
    },
    "Simulate a button click. Use `uiListEntries()` to find button names."
)

namespace
{
    template <typename T>
    Control::Value<T> readValue( const std::vector<std::string>& path )
    {
        Control::Value<T> ret;
        MR::CommandLoop::runCommandFromGUIThread( [&]
        {
            ret = MR::expectedValueOrThrow( Control::readValue<T>( path ) );
        } );
        return ret;
    }
}

MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiReadValueInt, readValue<std::int64_t>,
    "Read a value from a drag/slider widget. This function is for signed integers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiReadValueUint, readValue<std::uint64_t>,
    "Read a value from a drag/slider widget. This function is for unsigned integers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiReadValueReal, readValue<double>,
    "Read a value from a drag/slider widget. This function is for real numbers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiReadValueString, readValue<std::string>,
    "Read a value from a drag/slider widget. This function is for strings."
)

namespace
{
    template <typename T>
    void writeValue( const std::vector<std::string>& path, T value )
    {
        MR::CommandLoop::runCommandFromGUIThread( [&]
        {
            // Empty status = OK (write simulated); non-empty = disabled (silent no-op — pre-#5961 test contract).
            auto status = MR::expectedValueOrThrow( Control::writeValue<T>( path, std::move( value ) ) );
            if ( !status.empty() )
                spdlog::warn( "writeValue {}: {} (silent no-op)", Control::pathToString( path ), status );
        } );
    }
}

MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiWriteValue, writeValue<std::int64_t>,
    "Write a value to a drag/slider widget. This overload is for signed integers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiWriteValue, writeValue<std::uint64_t>,
    "Write a value to a drag/slider widget. This overload is for unsigned integers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiWriteValue, writeValue<double>,
    "Write a value to a drag/slider widget. This overload is for real numbers."
)
MR_ADD_PYTHON_FUNCTION( mrviewerpy, uiWriteValue, writeValue<std::string>,
    "Write a value to a drag/slider widget. This overload is for strings."
)

// ] end deprecated
