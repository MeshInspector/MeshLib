#include "MRWasmBindings.h"

#include "MRMesh/MRMeshComponents.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRVector.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <memory>
#include <vector>

using namespace MR;

namespace
{
struct MeshComponentsModule {};
}

EMSCRIPTEN_DECLARE_VAL_TYPE( FaceBitSetArrayVal )
EMSCRIPTEN_DECLARE_VAL_TYPE( ComponentsMapResultVal )
EMSCRIPTEN_DECLARE_VAL_TYPE( LargeByAreaRegionsResultVal )

EMSCRIPTEN_BINDINGS( meshlib_mesh_components )
{
    emscripten::register_type<FaceBitSetArrayVal>( "FaceBitSet[]" );
    emscripten::register_type<ComponentsMapResultVal>( "ComponentsMapResult",
        "{ map: Face2RegionMap; numRegions: number }" );
    emscripten::register_type<LargeByAreaRegionsResultVal>( "LargeByAreaRegionsResult",
        "{ faces: FaceBitSet; numRegions: number }" );

    emscripten::enum_<MeshComponents::FaceIncidence>( "MeshComponentsFaceIncidence" )
        .value( "PerEdge", MeshComponents::FaceIncidence::PerEdge )
        .value( "PerVertex", MeshComponents::FaceIncidence::PerVertex );

    emscripten::class_<MeshComponentsModule>( "MeshComponents" )
        .class_function( "getComponent", +[]( std::shared_ptr<Mesh> meshPart, int id, MeshComponents::FaceIncidence incidence )
        {
            return MeshComponents::getComponent( *meshPart, FaceId( id ), incidence );
        } )
        .class_function( "getComponents", +[]( std::shared_ptr<Mesh> meshPart, const FaceBitSet& seeds, MeshComponents::FaceIncidence incidence )
        {
            return MeshComponents::getComponents( *meshPart, seeds, incidence );
        } )
        .class_function( "getLargestComponent", +[]( std::shared_ptr<Mesh> meshPart, MeshComponents::FaceIncidence incidence, float minArea )
        {
            return MeshComponents::getLargestComponent( *meshPart, incidence, nullptr, minArea );
        } )
        .class_function( "getLargeByAreaComponents", +[]( std::shared_ptr<Mesh> mp, float minArea )
        {
            return MeshComponents::getLargeByAreaComponents( *mp, minArea, nullptr );
        } )
        .class_function( "getNumComponents", +[]( std::shared_ptr<Mesh> meshPart, MeshComponents::FaceIncidence incidence )
        {
            return MeshComponents::getNumComponents( *meshPart, incidence );
        } )
        .class_function( "getAllComponents", +[]( std::shared_ptr<Mesh> meshPart, MeshComponents::FaceIncidence incidence )
        {
            const std::vector<FaceBitSet> comps = MeshComponents::getAllComponents( *meshPart, incidence );
            emscripten::val arr = emscripten::val::array();
            for ( const FaceBitSet& c : comps )
                arr.call<void>( "push", c );
            return FaceBitSetArrayVal( arr );
        } )
        .class_function( "getAllComponentsMap", +[]( std::shared_ptr<Mesh> meshPart, MeshComponents::FaceIncidence incidence )
        {
            auto [regionMap, numRegions] = MeshComponents::getAllComponentsMap( *meshPart, incidence );
            emscripten::val out = emscripten::val::object();
            out.set( "map", regionMap );
            out.set( "numRegions", numRegions );
            return ComponentsMapResultVal( out );
        } )
        .class_function( "getLargeByAreaRegions", +[]( std::shared_ptr<Mesh> meshPart, const Face2RegionMap& regionMap, int numRegions, float minArea )
        {
            auto [faces, n] = MeshComponents::getLargeByAreaRegions( *meshPart, regionMap, numRegions, minArea );
            emscripten::val out = emscripten::val::object();
            out.set( "faces", faces );
            out.set( "numRegions", n );
            return LargeByAreaRegionsResultVal( out );
        } );
}
