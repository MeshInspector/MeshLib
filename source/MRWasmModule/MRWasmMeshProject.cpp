#include "MRWasmBindings.h"

#include "MRMesh/MRMeshProject.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <memory>
#include <optional>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh_project )
{
    emscripten::function( "findProjection", +[]( const Vector3f& pt, std::shared_ptr<Mesh> mp )
    {
        const MeshProjectionResult r = findProjection( pt, *mp );

        emscripten::val proj = emscripten::val::object();
        proj.set( "face", (int)r.proj.face );
        proj.set( "point", r.proj.point );

        emscripten::val bary = emscripten::val::object();
        bary.set( "a", r.mtp.bary.a );
        bary.set( "b", r.mtp.bary.b );
        emscripten::val mtp = emscripten::val::object();
        mtp.set( "e", (int)r.mtp.e );
        mtp.set( "bary", bary );

        emscripten::val out = emscripten::val::object();
        out.set( "proj", proj );
        out.set( "mtp", mtp );
        out.set( "distSq", r.distSq );
        out.set( "valid", r.valid() );
        return out;
    } );

    emscripten::function( "findSignedDistanceFromPoint", +[]( const Vector3f& pt, std::shared_ptr<Mesh> mp )
    {
        const auto res = findSignedDistance( pt, *mp );
        if ( !res )
            return emscripten::val::null();

        emscripten::val proj = emscripten::val::object();
        proj.set( "face", (int)res->proj.face );
        proj.set( "point", res->proj.point );

        emscripten::val bary = emscripten::val::object();
        bary.set( "a", res->mtp.bary.a );
        bary.set( "b", res->mtp.bary.b );
        emscripten::val mtp = emscripten::val::object();
        mtp.set( "e", (int)res->mtp.e );
        mtp.set( "bary", bary );

        emscripten::val out = emscripten::val::object();
        out.set( "proj", proj );
        out.set( "mtp", mtp );
        out.set( "dist", res->dist );
        return out;
    } );
}
