#include "MRWasmBindings.h"

#include "MRMesh/MRContoursCut.h"
#include "MRMesh/MROneMeshContours.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRVector3.h"

#include <emscripten/bind.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_contours_cut )
{
    emscripten::enum_<CutMeshParameters::FillPart>( "FillPart" )
        .value( "Both", CutMeshParameters::FillPart::Both )
        .value( "Left", CutMeshParameters::FillPart::Left )
        .value( "Right", CutMeshParameters::FillPart::Right );

    emscripten::enum_<CutMeshParameters::ForceFill>( "ForceFill" )
        .value( "None", CutMeshParameters::ForceFill::None )
        .value( "Good", CutMeshParameters::ForceFill::Good )
        .value( "All", CutMeshParameters::ForceFill::All );

    emscripten::class_<CutMeshParameters>( "CutMeshParameters" )
        .constructor<>()
        .property( "fillPart", &CutMeshParameters::fillPart )
        .property( "forceFillMode", &CutMeshParameters::forceFillMode );

    emscripten::function( "cutMesh", +[]( std::shared_ptr<Mesh> mesh, const OneMeshContours& contours, const CutMeshParameters& params )
    {
        auto res = cutMesh( *mesh, contours, params );
        auto resultCut = emscripten::val::array();
        for ( const auto& path : res.resultCut )
            resultCut.call<void>( "push", Wasm::packedToTypedArray<EdgePath, uint32_t>( path ) );
        auto out = emscripten::val::object();
        out.set( "resultCut", resultCut );
        out.set( "fbsWithContourIntersections", res.fbsWithContourIntersections );
        return out;
    } );

    emscripten::class_<CutByProjectionSettings>( "CutByProjectionSettings" )
        .constructor<>()
        .property( "direction", &CutByProjectionSettings::direction );

    emscripten::function( "cutMeshByProjection", +[]( std::shared_ptr<Mesh> mesh, emscripten::val contours, const CutByProjectionSettings& settings )
    {
        Contours3f cppContours;
        const size_t n = contours[ "length" ].as<size_t>();
        cppContours.resize( n );
        for ( size_t i = 0; i < n; ++i )
        {
            emscripten::val arr = contours[ i ];
            const size_t len = arr[ "length" ].as<size_t>();
            cppContours[ i ].resize( len / 3 );
            if ( len != 0 )
            {
                emscripten::val view( emscripten::typed_memory_view( len, reinterpret_cast<float*>( cppContours[ i ].data() ) ) );
                view.call<void>( "set", arr );
            }
        }
        auto paths = Wasm::unwrap( cutMeshByProjection( *mesh, cppContours, settings ) );
        auto out = emscripten::val::array();
        for ( const auto& path : paths )
            out.call<void>( "push", Wasm::packedToTypedArray<EdgePath, uint32_t>( path ) );
        return out;
    } );
}
