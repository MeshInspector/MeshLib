#include "MRPython/MRPython.h"
#include "MRMesh/MRMovementBuildBody.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMesh.h"

// for automatic conversion of std::optional into python type,
// but it also affects all std::vector's making them another python type not like in other translation units
#include <pybind11/stl.h>

using namespace MR;

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, MovementBuildBodyParams, MovementBuildBodyParams )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MovementBuildBodyParams, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( MovementBuildBodyParams ).
        def( pybind11::init<>() ).
        def_readwrite( "allowRotation", &MovementBuildBodyParams::allowRotation,
            "if this flag is set, rotate body in trajectory vertices\n"
            "according to its rotation\n"
            "otherwise body movement will be done without any rotation" ).
        def_readwrite( "center", &MovementBuildBodyParams::center,
            "point in body space that follows trajectory\n"
            "if not set body bounding box center is used" ).
        def_readwrite( "bodyNormal", &MovementBuildBodyParams::bodyNormal,
            "facing direction of body, used for initial rotation (if allowRotation)\n"
            "if not set body accumulative normal is used" ).
        def_readwrite( "b2tXf", &MovementBuildBodyParams::b2tXf, "optional transform body space to trajectory space" );
} )
