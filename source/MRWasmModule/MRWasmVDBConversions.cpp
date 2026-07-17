#include "MRWasmBindings.h"

#include "MRVoxels/MRVDBConversions.h"
#include "MRVoxels/MRVoxelsVolume.h"
#include "MRVoxels/MRFloatGrid.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_vdb_conversions )
{
    emscripten::enum_<MeshToVolumeParams::Type>( "MeshToVolumeType" )
        .value( "Signed", MeshToVolumeParams::Type::Signed )
        .value( "Unsigned", MeshToVolumeParams::Type::Unsigned );

    emscripten::class_<MeshToVolumeParams>( "MeshToVolumeParams" )
        .constructor<>()
        .property( "type", &MeshToVolumeParams::type )
        .property( "surfaceOffset", &MeshToVolumeParams::surfaceOffset )
        .property( "voxelSize", &MeshToVolumeParams::voxelSize )
        .property( "worldXf", &MeshToVolumeParams::worldXf );

    emscripten::class_<GridToMeshSettings>( "GridToMeshSettings" )
        .constructor<>()
        .property( "voxelSize", &GridToMeshSettings::voxelSize )
        .property( "isoValue", &GridToMeshSettings::isoValue )
        .property( "adaptivity", &GridToMeshSettings::adaptivity )
        .property( "maxFaces", &GridToMeshSettings::maxFaces )
        .property( "maxVertices", &GridToMeshSettings::maxVertices )
        .property( "relaxDisorientedTriangles", &GridToMeshSettings::relaxDisorientedTriangles );

    emscripten::function( "evalGridMinMax", +[]( const FloatGrid& grid )
    {
        float mn = 0, mx = 0;
        evalGridMinMax( grid, mn, mx );
        auto out = emscripten::val::object();
        out.set( "min", mn );
        out.set( "max", mx );
        return out;
    } );

    emscripten::function( "meshToVolume", +[]( std::shared_ptr<Mesh> mp, const MeshToVolumeParams& params )
    {
        return Wasm::unwrap( meshToVolume( *mp, params ) );
    } );
    emscripten::function( "meshToDistanceVdbVolume", +[]( std::shared_ptr<Mesh> mp, const MeshToVolumeParams& params )
    {
        return Wasm::unwrap( meshToDistanceVdbVolume( *mp, params ) );
    } );
    emscripten::function( "floatGridToVdbVolume", +[]( FloatGrid grid )
    {
        return floatGridToVdbVolume( grid );
    } );
    emscripten::function( "gridToMesh", +[]( const FloatGrid& grid, const GridToMeshSettings& settings )
    {
        return std::make_shared<Mesh>( Wasm::unwrap( gridToMesh( grid, settings ) ) );
    } );
}
