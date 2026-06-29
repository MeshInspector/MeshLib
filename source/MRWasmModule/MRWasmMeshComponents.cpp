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

EMSCRIPTEN_BINDINGS( meshlib_mesh_components )
{
    emscripten::enum_<MeshComponents::FaceIncidence>( "MeshComponentsFaceIncidence" )
        .value( "PerEdge", MeshComponents::FaceIncidence::PerEdge )
        .value( "PerVertex", MeshComponents::FaceIncidence::PerVertex );

    emscripten::class_<MeshComponentsModule>( "MeshComponents" )
        .class_function( "getComponent", +[]( std::shared_ptr<Mesh> m, int seed, MeshComponents::FaceIncidence inc )
        {
            return MeshComponents::getComponent( *m, FaceId( seed ), inc );
        } )
        .class_function( "getComponents", +[]( std::shared_ptr<Mesh> m, const FaceBitSet& seeds, MeshComponents::FaceIncidence inc )
        {
            return MeshComponents::getComponents( *m, seeds, inc );
        } )
        .class_function( "getLargestComponent", +[]( std::shared_ptr<Mesh> m, MeshComponents::FaceIncidence inc, float minArea )
        {
            return MeshComponents::getLargestComponent( *m, inc, nullptr, minArea );
        } )
        .class_function( "getLargeByAreaComponents", +[]( std::shared_ptr<Mesh> m, float minArea )
        {
            return MeshComponents::getLargeByAreaComponents( *m, minArea, nullptr );
        } )
        .class_function( "getNumComponents", +[]( std::shared_ptr<Mesh> m, MeshComponents::FaceIncidence inc )
        {
            return MeshComponents::getNumComponents( *m, inc );
        } )
        .class_function( "getAllComponents", +[]( std::shared_ptr<Mesh> m, MeshComponents::FaceIncidence inc )
        {
            const std::vector<FaceBitSet> comps = MeshComponents::getAllComponents( *m, inc );
            emscripten::val arr = emscripten::val::array();
            for ( const FaceBitSet& c : comps )
                arr.call<void>( "push", c );
            return arr;
        } )
        .class_function( "getAllComponentsMap", +[]( std::shared_ptr<Mesh> m, MeshComponents::FaceIncidence inc )
        {
            auto [regionMap, numRegions] = MeshComponents::getAllComponentsMap( *m, inc );
            emscripten::val out = emscripten::val::object();
            out.set( "map", regionMap );
            out.set( "numRegions", numRegions );
            return out;
        } )
        .class_function( "getLargeByAreaRegions", +[]( std::shared_ptr<Mesh> m, const Face2RegionMap& face2RegionMap, int numRegions, float minArea )
        {
            auto [faces, n] = MeshComponents::getLargeByAreaRegions( *m, face2RegionMap, numRegions, minArea );
            emscripten::val out = emscripten::val::object();
            out.set( "faces", faces );
            out.set( "numRegions", n );
            return out;
        } );
}
