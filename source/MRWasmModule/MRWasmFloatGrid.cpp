#include "MRWasmBindings.h"

#include "MRVoxels/MRFloatGrid.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRBox.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_float_grid )
{
    emscripten::class_<FloatGrid>( "FloatGrid" )
        .constructor<>();

    emscripten::function( "resampled", +[]( const FloatGrid& grid, float voxelScale )
    {
        return resampled( grid, voxelScale );
    } );
    emscripten::function( "resampled", +[]( const FloatGrid& grid, float scaleX, float scaleY, float scaleZ )
    {
        return resampled( grid, Vector3f( scaleX, scaleY, scaleZ ) );
    } );
    emscripten::function( "cropped", +[]( const FloatGrid& grid, const Box3i& box )
    {
        return cropped( grid, box );
    } );
    emscripten::function( "getValue", +[]( const FloatGrid& grid, const Vector3i& p )
    {
        return getValue( grid, p );
    } );
    emscripten::function( "setValue", +[]( FloatGrid& grid, const Vector3i& p, float value )
    {
        setValue( grid, p, value );
    } );
}
