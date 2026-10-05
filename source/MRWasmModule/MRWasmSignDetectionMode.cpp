#include "MRWasmBindings.h"

#include "MRMesh/MRSignDetectionMode.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_sign_detection_mode )
{
    emscripten::enum_<SignDetectionMode>( "SignDetectionMode" )
        .value( "Unsigned", SignDetectionMode::Unsigned )
        .value( "OpenVDB", SignDetectionMode::OpenVDB )
        .value( "ProjectionNormal", SignDetectionMode::ProjectionNormal )
        .value( "WindingRule", SignDetectionMode::WindingRule ) // old name of OddCrossings, which is registered after it to be returned from C++
        .value( "OddCrossings", SignDetectionMode::OddCrossings )
        .value( "HoleWindingRule", SignDetectionMode::HoleWindingRule );
}
