#include "MRPython/MRPython.h"

#include <MRMesh/MRSystemPath.h>

#pragma warning(push)
#pragma warning(disable: 4464) // relative include path contains '..'
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#pragma warning(pop)

using namespace MR;

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SystemPath, [] ( pybind11::module_& m )
{
    auto clsSystemPath = pybind11::class_<SystemPath>( m, "SystemPath" );

    pybind11::enum_<SystemPath::Directory>( clsSystemPath, "Directory" )
        .value( "Resources", SystemPath::Directory::Resources )
        .value( "Fonts", SystemPath::Directory::Fonts )
        .value( "Plugins", SystemPath::Directory::Plugins )
        .value( "PythonModules", SystemPath::Directory::PythonModules )
    ;

    clsSystemPath
        .def_static( "getDirectory", &SystemPath::getDirectory,
            pybind11::arg( "dir" ),
            "get the directory path for specified category"
        )
        .def_static( "overrideDirectory", &SystemPath::overrideDirectory,
            pybind11::arg( "dir" ), pybind11::arg( "path" ),
            "override the directory path for specified category, useful for custom configurations"
        )
    ;
} )
