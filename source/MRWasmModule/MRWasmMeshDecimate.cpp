#include "MRWasmBindings.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRMeshDecimate.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

using namespace emscripten;

namespace
{

struct DecimateResultJs
{
    int vertsDeleted = 0;
    int facesDeleted = 0;
    float errorIntroduced = 0;
    bool cancelled = false;
};

DecimateResultJs decimate( MR::Mesh& mesh, val settings )
{
    return guarded( [&]() -> DecimateResultJs
    {
        MR::DecimateSettings s;
        s.packMesh = true;

        const int curFaces = mesh.topology.numValidFaces();
        const bool hasSettings = !settings.isUndefined() && !settings.isNull();
        auto has = [&]( const char* k ) { return hasSettings && settings.hasOwnProperty( k ); };
        auto num = [&]( const char* k ) { return settings[k].as<double>(); };

        bool haveStop = false;

        if ( has( "targetTriangleCount" ) )
        {
            int target = (int)num( "targetTriangleCount" );
            if ( target < 0 )
                target = 0;
            s.maxDeletedFaces = std::max( 0, curFaces - target );
            haveStop = true;
        }
        else if ( has( "targetRatio" ) )
        {
            const double r = std::clamp( num( "targetRatio" ), 0.0, 1.0 );
            const int target = (int)std::llround( curFaces * r );
            s.maxDeletedFaces = std::max( 0, curFaces - target );
            haveStop = true;
        }

        if ( has( "maxDeletedFaces" ) ) { s.maxDeletedFaces = (int)num( "maxDeletedFaces" ); haveStop = true; }
        if ( has( "maxError" ) ) { s.maxError = (float)num( "maxError" ); haveStop = true; }
        if ( has( "maxEdgeLen" ) ) { s.maxEdgeLen = (float)num( "maxEdgeLen" ); haveStop = true; }
        if ( has( "strategy" ) )
        {
            const std::string st = settings["strategy"].as<std::string>();
            s.strategy = ( st == "shortestEdgeFirst" )
                ? MR::DecimateStrategy::ShortestEdgeFirst
                : MR::DecimateStrategy::MinimizeError;
        }

        if ( !haveStop )
            throw std::runtime_error( "decimate: provide a stopping criterion "
                "(targetRatio, targetTriangleCount, maxDeletedFaces, maxError or maxEdgeLen)" );

        const MR::DecimateResult r = MR::decimateMesh( mesh, s );
        return DecimateResultJs{ r.vertsDeleted, r.facesDeleted, r.errorIntroduced, r.cancelled };
    } );
}

}

EMSCRIPTEN_BINDINGS( meshlib_decimate )
{
    value_object<DecimateResultJs>( "DecimateResult" )
        .field( "vertsDeleted", &DecimateResultJs::vertsDeleted )
        .field( "facesDeleted", &DecimateResultJs::facesDeleted )
        .field( "errorIntroduced", &DecimateResultJs::errorIntroduced )
        .field( "cancelled", &DecimateResultJs::cancelled );

    function( "decimate", &decimate );
}
