#include "MRWasmBindings.h"

#include "MRVoxels/MRVoxelsVolume.h"
#include "MRVoxels/MRFloatGrid.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_vdb_volume )
{
    emscripten::class_<VdbVolume>( "VdbVolume" )
        .property( "data", +[]( const VdbVolume& v ) { return v.data; } )
        .property( "dims", +[]( const VdbVolume& v ) { return v.dims; } )
        .property( "voxelSize", +[]( const VdbVolume& v ) { return v.voxelSize; } )
        .property( "min", +[]( const VdbVolume& v ) { return v.min; } )
        .property( "max", +[]( const VdbVolume& v ) { return v.max; } );
}
