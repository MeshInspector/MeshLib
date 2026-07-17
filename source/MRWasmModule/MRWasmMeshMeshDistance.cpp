#include "MRWasmBindings.h"

#include "MRMesh/MRMeshMeshDistance.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_mesh_distance )
{
    emscripten::enum_<MeshMeshCollisionStatus>( "MeshMeshCollisionStatus" )
        .value( "BothOutside", MeshMeshCollisionStatus::BothOutside )
        .value( "BothInside", MeshMeshCollisionStatus::BothInside )
        .value( "AInside", MeshMeshCollisionStatus::AInside )
        .value( "BInside", MeshMeshCollisionStatus::BInside )
        .value( "Colliding", MeshMeshCollisionStatus::Colliding )
        .value( "Touching", MeshMeshCollisionStatus::Touching )
        .value( "NotColliding", MeshMeshCollisionStatus::NotColliding );

    emscripten::function( "findDistance", +[]( std::shared_ptr<Mesh> a, std::shared_ptr<Mesh> b )
    {
        const MeshMeshDistanceResult r = findDistance( *a, *b );

        emscripten::val va = emscripten::val::object();
        va.set( "face", (int)r.a.face );
        va.set( "point", r.a.point );
        emscripten::val vb = emscripten::val::object();
        vb.set( "face", (int)r.b.face );
        vb.set( "point", r.b.point );

        emscripten::val out = emscripten::val::object();
        out.set( "a", va );
        out.set( "b", vb );
        out.set( "distSq", r.distSq );
        return out;
    } );

    emscripten::function( "findSignedDistanceFromMesh", +[]( std::shared_ptr<Mesh> a, std::shared_ptr<Mesh> b )
    {
        const MeshMeshSignedDistanceResult r = findSignedDistance( *a, *b );

        emscripten::val va = emscripten::val::object();
        va.set( "face", (int)r.a.face );
        va.set( "point", r.a.point );
        emscripten::val vb = emscripten::val::object();
        vb.set( "face", (int)r.b.face );
        vb.set( "point", r.b.point );

        emscripten::val out = emscripten::val::object();
        out.set( "a", va );
        out.set( "b", vb );
        out.set( "status", (int)r.status );
        out.set( "signedDist", r.signedDist );
        return out;
    } );
}
