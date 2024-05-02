#include "MRMesh/MRPython.h"
#include "MRMesh/MRMovementBuildBody.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMesh.h"
#include <pybind11/stl.h> // for automatic conversion of std::optional into python type

using namespace MR;

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MovementBody, [] ( pybind11::module_& m )
{
    pybind11::class_<MovementBuildBodyParams>( m, "MovementBuildBodyParams" ).
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

    m.def( "makeMovementBuildBody", &makeMovementBuildBody,
        pybind11::arg( "body" ), pybind11::arg( "trajectory" ), pybind11::arg_v( "params", MovementBuildBodyParams(), "MovementBuildBodyParams()" ),
        "makes mesh by moving `body` along `trajectory`\n"
        "if allowRotation rotate it in corners" );
} )

