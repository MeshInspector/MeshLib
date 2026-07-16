#include "MRWasmBindings.h"

#include "MRVoxels/MROffset.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRSignDetectionMode.h"
#include "MRMesh/MREnums.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_offset )
{
    emscripten::class_<BaseShellParameters>( "BaseShellParameters" )
        .constructor<>()
        .property( "voxelSize", &BaseShellParameters::voxelSize );

    emscripten::class_<OffsetParameters, emscripten::base<BaseShellParameters>>( "OffsetParameters" )
        .constructor<>()
        .property( "signDetectionMode", &OffsetParameters::signDetectionMode )
        .property( "memoryEfficient", &OffsetParameters::memoryEfficient );

    emscripten::class_<SharpOffsetParameters, emscripten::base<OffsetParameters>>( "SharpOffsetParameters" )
        .constructor<>()
        .property( "minNewVertDev", &SharpOffsetParameters::minNewVertDev )
        .property( "maxNewRank2VertDev", &SharpOffsetParameters::maxNewRank2VertDev )
        .property( "maxNewRank3VertDev", &SharpOffsetParameters::maxNewRank3VertDev )
        .property( "maxOldVertPosCorrection", &SharpOffsetParameters::maxOldVertPosCorrection );

    emscripten::class_<GeneralOffsetParameters, emscripten::base<SharpOffsetParameters>>( "GeneralOffsetParameters" )
        .constructor<>()
        .property( "mode", &GeneralOffsetParameters::mode );

    emscripten::function( "suggestVoxelSize", +[]( std::shared_ptr<Mesh> mp, float approxNumVoxels )
    {
        return suggestVoxelSize( *mp, approxNumVoxels );
    } );
    emscripten::function( "offsetMesh", +[]( std::shared_ptr<Mesh> mp, float offset, const OffsetParameters& params )
    {
        return std::make_shared<Mesh>( Wasm::unwrap( offsetMesh( *mp, offset, params ) ) );
    } );
    emscripten::function( "doubleOffsetMesh", +[]( std::shared_ptr<Mesh> mp, float offsetA, float offsetB, const OffsetParameters& params )
    {
        return std::make_shared<Mesh>( Wasm::unwrap( doubleOffsetMesh( *mp, offsetA, offsetB, params ) ) );
    } );
    emscripten::function( "mcOffsetMesh", +[]( std::shared_ptr<Mesh> mp, float offset, const OffsetParameters& params )
    {
        return std::make_shared<Mesh>( Wasm::unwrap( mcOffsetMesh( *mp, offset, params ) ) );
    } );
    emscripten::function( "mcShellMeshRegion", +[]( std::shared_ptr<Mesh> mesh, const FaceBitSet& region, float offset, const BaseShellParameters& params )
    {
        return std::make_shared<Mesh>( Wasm::unwrap( mcShellMeshRegion( *mesh, region, offset, params ) ) );
    } );
    emscripten::function( "sharpOffsetMesh", +[]( std::shared_ptr<Mesh> mp, float offset, const SharpOffsetParameters& params )
    {
        return std::make_shared<Mesh>( Wasm::unwrap( sharpOffsetMesh( *mp, offset, params ) ) );
    } );
    emscripten::function( "generalOffsetMesh", +[]( std::shared_ptr<Mesh> mp, float offset, const GeneralOffsetParameters& params )
    {
        return std::make_shared<Mesh>( Wasm::unwrap( generalOffsetMesh( *mp, offset, params ) ) );
    } );
    emscripten::function( "thickenMesh", +[]( std::shared_ptr<Mesh> mesh, float offset, const GeneralOffsetParameters& params )
    {
        return std::make_shared<Mesh>( Wasm::unwrap( thickenMesh( *mesh, offset, params ) ) );
    } );
}
