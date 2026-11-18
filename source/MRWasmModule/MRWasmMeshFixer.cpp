#include "MRWasmBindings.h"

#include "MRMesh/MRMeshFixer.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_fixer )
{
    emscripten::enum_<FixMeshDegeneraciesParams::Mode>( "FixMeshDegeneraciesMode" )
        .value( "Decimate", FixMeshDegeneraciesParams::Mode::Decimate )
        .value( "Remesh", FixMeshDegeneraciesParams::Mode::Remesh )
        .value( "RemeshPatch", FixMeshDegeneraciesParams::Mode::RemeshPatch );

    emscripten::class_<FixMeshDegeneraciesParams>( "FixMeshDegeneraciesParams" )
        .constructor<>()
        .property( "maxDeviation", &FixMeshDegeneraciesParams::maxDeviation )
        .property( "tinyEdgeLength", &FixMeshDegeneraciesParams::tinyEdgeLength )
        .property( "criticalTriAspectRatio", &FixMeshDegeneraciesParams::criticalTriAspectRatio )
        .property( "maxAngleChange", &FixMeshDegeneraciesParams::maxAngleChange )
        .property( "stabilizer", &FixMeshDegeneraciesParams::stabilizer )
        .property( "mode", &FixMeshDegeneraciesParams::mode )
        .property( "mimicPatch", &FixMeshDegeneraciesParams::mimicPatch );

    emscripten::function( "fixMeshDegeneracies",
        +[]( std::shared_ptr<Mesh> mesh, const FixMeshDegeneraciesParams& params )
    {
        Wasm::unwrap( fixMeshDegeneracies( *mesh, params ) );
    } );

    emscripten::function( "fixMultipleEdges", +[]( std::shared_ptr<Mesh> mesh )
    {
        fixMultipleEdges( *mesh );
    } );

    emscripten::function( "findDegenerateFaces", +[]( std::shared_ptr<Mesh> mp, float criticalAspectRatio )
    {
        return Wasm::unwrap( findDegenerateFaces( *mp, criticalAspectRatio ) );
    } );

    emscripten::function( "findShortEdges", +[]( std::shared_ptr<Mesh> mp, float criticalLength )
    {
        return Wasm::unwrap( findShortEdges( *mp, criticalLength ) );
    } );

    emscripten::function( "findHoleComplicatingFaces", +[]( std::shared_ptr<Mesh> mesh )
    {
        return findHoleComplicatingFaces( *mesh );
    } );
}
