#include "MRWasmBindings.h"

#include "MRMesh/MRMultiwayICP.h"
#include "MRMesh/MRICP.h"
#include "MRMesh/MRMeshOrPoints.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRVector.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace MR;

EMSCRIPTEN_DECLARE_VAL_TYPE( AffineXf3fArrayVal )
EMSCRIPTEN_DECLARE_VAL_TYPE( MeshOrPointsXfArrayVal )

EMSCRIPTEN_BINDINGS( meshlib_multiway_icp )
{
    emscripten::register_type<AffineXf3fArrayVal>( "AffineXf3f[]" );
    emscripten::register_type<MeshOrPointsXfArrayVal>( "readonly MeshOrPointsXf[]" );

    emscripten::enum_<MultiwayICPSamplingParameters::CascadeMode>( "CascadeMode" )
        .value( "Sequential", MultiwayICPSamplingParameters::CascadeMode::Sequential )
        .value( "AABBTreeBased", MultiwayICPSamplingParameters::CascadeMode::AABBTreeBased );

    emscripten::class_<MultiwayICPSamplingParameters>( "MultiwayICPSamplingParameters" )
        .constructor<>()
        .property( "samplingVoxelSize", &MultiwayICPSamplingParameters::samplingVoxelSize )
        .property( "maxGroupSize", &MultiwayICPSamplingParameters::maxGroupSize )
        .property( "cascadeMode", &MultiwayICPSamplingParameters::cascadeMode );

    emscripten::class_<MultiwayICP>( "MultiwayICP" )
        .constructor( +[]( MeshOrPointsXfArrayVal objects, const MultiwayICPSamplingParameters& params )
        {
            ICPObjects objs;
            const size_t n = objects[ "length" ].as<size_t>();
            objs.reserve( n );
            for ( size_t i = 0; i < n; ++i )
                objs.push_back( objects[ i ].as<MeshOrPointsXf>() );
            return new MultiwayICP( objs, params );
        }, emscripten::allow_raw_pointers() )
        .function( "calculateTransformations", +[]( MultiwayICP& self )
        {
            const auto xfs = self.calculateTransformations();
            emscripten::val arr = emscripten::val::array();
            for ( const auto& xf : xfs )
                arr.call<void>( "push", xf );
            return AffineXf3fArrayVal( arr );
        } )
        .function( "resamplePoints", &MultiwayICP::resamplePoints )
        .function( "updateAllPointPairs", +[]( MultiwayICP& self ) { return self.updateAllPointPairs(); } )
        .function( "setParams", &MultiwayICP::setParams )
        .function( "getParams", +[]( const MultiwayICP& self ) { return self.getParams(); } )
        .function( "getMeanSqDistToPoint", +[]( const MultiwayICP& self ) { return self.getMeanSqDistToPoint(); } )
        .function( "getMeanSqDistToPlane", +[]( const MultiwayICP& self ) { return self.getMeanSqDistToPlane(); } )
        .function( "getNumSamples", &MultiwayICP::getNumSamples )
        .function( "getNumActivePairs", &MultiwayICP::getNumActivePairs )
        .function( "getStatusInfo", &MultiwayICP::getStatusInfo );
}
