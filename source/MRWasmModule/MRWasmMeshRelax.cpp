#include "MRWasmBindings.h"

#include "MRMesh/MRMeshRelax.h"
#include "MRMesh/MRMesh.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_relax )
{
    emscripten::class_<MeshRelaxParams>( "MeshRelaxParams" )
        .constructor<>()
        .property( "iterations", static_cast<int MeshRelaxParams::*>( &MeshRelaxParams::iterations ) )
        .property( "force", static_cast<float MeshRelaxParams::*>( &MeshRelaxParams::force ) )
        .property( "limitNearInitial", static_cast<bool MeshRelaxParams::*>( &MeshRelaxParams::limitNearInitial ) )
        .property( "maxInitialDist", static_cast<float MeshRelaxParams::*>( &MeshRelaxParams::maxInitialDist ) )
        .property( "hardSmoothTetrahedrons", &MeshRelaxParams::hardSmoothTetrahedrons );

    emscripten::function( "relax", +[]( std::shared_ptr<Mesh> m, const MeshRelaxParams& p )
    {
        return relax( *m, p );
    } );

    emscripten::function( "relaxKeepVolume", +[]( std::shared_ptr<Mesh> m, const MeshRelaxParams& p )
    {
        return relaxKeepVolume( *m, p );
    } );
}
