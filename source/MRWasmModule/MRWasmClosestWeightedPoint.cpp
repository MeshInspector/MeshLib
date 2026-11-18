#include "MRWasmBindings.h"

#include "MRMesh/MRClosestWeightedPoint.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_closest_weighted_point )
{
    emscripten::class_<DistanceFromWeightedPointsParams>( "DistanceFromWeightedPointsParams" )
        .constructor<>()
        .property( "minWeight", &DistanceFromWeightedPointsParams::minWeight )
        .property( "maxWeight", &DistanceFromWeightedPointsParams::maxWeight )
        .property( "maxWeightGrad", &DistanceFromWeightedPointsParams::maxWeightGrad )
        .property( "bidirectionalMode", &DistanceFromWeightedPointsParams::bidirectionalMode );
}
