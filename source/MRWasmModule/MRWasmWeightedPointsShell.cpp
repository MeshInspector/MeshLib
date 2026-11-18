#include "MRWasmBindings.h"

#include "MRVoxels/MRWeightedPointsShell.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

namespace
{
struct WeightedShellModule {};
}

EMSCRIPTEN_BINDINGS( meshlib_weighted_points_shell )
{
    emscripten::class_<WeightedShell::ParametersBase>( "WeightedShellParametersBase" )
        .constructor<>()
        .property( "offset", &WeightedShell::ParametersBase::offset )
        .property( "voxelSize", &WeightedShell::ParametersBase::voxelSize )
        .property( "numLayers", &WeightedShell::ParametersBase::numLayers );

    emscripten::class_<WeightedShell::ParametersMetric, emscripten::base<WeightedShell::ParametersBase>>( "WeightedShellParametersMetric" )
        .constructor<>()
        .property( "dist", &WeightedShell::ParametersMetric::dist );

    emscripten::class_<WeightedShellModule>( "WeightedShell" )
        .class_function( "meshShell", +[]( std::shared_ptr<Mesh> mesh, const VertScalars& vertWeights, const WeightedShell::ParametersMetric& params )
        {
            return std::make_shared<Mesh>( Wasm::unwrap( WeightedShell::meshShell( *mesh, vertWeights, params ) ) );
        } );
}
