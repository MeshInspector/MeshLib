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
        +[]( std::shared_ptr<Mesh> m, const FixMeshDegeneraciesParams& p )
    {
        Wasm::unwrap( fixMeshDegeneracies( *m, p ) );
    } );

    emscripten::function( "fixMultipleEdges", +[]( std::shared_ptr<Mesh> m )
    {
        fixMultipleEdges( *m );
    } );

    emscripten::function( "findDegenerateFaces", +[]( std::shared_ptr<Mesh> m, float criticalAspectRatio )
    {
        return Wasm::unwrap( findDegenerateFaces( *m, criticalAspectRatio ) );
    } );

    emscripten::function( "findShortEdges", +[]( std::shared_ptr<Mesh> m, float criticalLength )
    {
        return Wasm::unwrap( findShortEdges( *m, criticalLength ) );
    } );

    emscripten::function( "findHoleComplicatingFaces", +[]( std::shared_ptr<Mesh> m )
    {
        return findHoleComplicatingFaces( *m );
    } );
}
