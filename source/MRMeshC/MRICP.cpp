#include "MRICP.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRICP.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( ICP )
REGISTER_AUTO_CAST( ICPMethod )
REGISTER_AUTO_CAST( ICPMode )
REGISTER_AUTO_CAST( ICPPairData )
REGISTER_AUTO_CAST( IPointPairs )
REGISTER_AUTO_CAST( MeshOrPointsXf )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST( VertBitSet )
REGISTER_AUTO_CAST2( std::string, MRString )

static_assert( sizeof( MRICPPairData ) == sizeof( MR::ICPPairData ) );

static_assert( sizeof( MRPointPair ) == sizeof( MR::PointPair ) );

#define COPY_FROM( obj, field ) . field = ( obj ). field ,

const MRICPPairData* mrIPointPairsGet( const MRIPointPairs* pp_, size_t idx )
{
    ARG( pp );
    RETURN( &pp[idx] );
}

size_t mrIPointPairsSize( const MRIPointPairs* pp_ )
{
    ARG( pp );
    RETURN( pp.size() );
}

MRICPPairData* mrIPointPairsGetRef( MRIPointPairs* pp_, size_t idx )
{
    ARG( pp );
    RETURN( &pp[idx] );
}

MRICPProperties mrICPPropertiesNew( void )
{
    static const ICPProperties def;
    return {
        .method = auto_cast( def.method ),
        COPY_FROM( def, p2plAngleLimit )
        COPY_FROM( def, p2plScaleLimit )
        COPY_FROM( def, cosThreshold )
        COPY_FROM( def, distThresholdSq )
        COPY_FROM( def, farDistFactor )
        .icpMode = auto_cast( def.icpMode ),
        .fixedRotationAxis = auto_cast( def.fixedRotationAxis ),
        COPY_FROM( def, iterLimit )
        COPY_FROM( def, badIterStopCount )
        COPY_FROM( def, exitVal )
        COPY_FROM( def, mutualClosest )
    };
}

MRICP* mrICPNew( const MRMeshOrPointsXf* flt_, const MRMeshOrPointsXf* ref_, float samplingVoxelSize )
{
    ARG( flt ); ARG( ref );
    RETURN_NEW( ICP( flt, ref, samplingVoxelSize ) );
}

MRICP* mrICPNewFromSamples( const MRMeshOrPointsXf* flt_, const MRMeshOrPointsXf* ref_, const MRVertBitSet* fltSamples_, const MRVertBitSet* refSamples_ )
{
    ARG( flt ); ARG( ref ); ARG_PTR( fltSamples ); ARG_PTR( refSamples );
    RETURN_NEW( ICP( flt, ref, *fltSamples, *refSamples ) );
}


void mrICPSetParams( MRICP* icp_, const MRICPProperties* prop_ )
{
    ARG( icp );
    const ICPProperties prop {
        .method = auto_cast( prop_->method ),
        COPY_FROM( *prop_, p2plAngleLimit )
        COPY_FROM( *prop_, p2plScaleLimit )
        COPY_FROM( *prop_, cosThreshold )
        COPY_FROM( *prop_, distThresholdSq )
        COPY_FROM( *prop_, farDistFactor )
        .icpMode = auto_cast( prop_->icpMode ),
        .fixedRotationAxis = auto_cast( prop_->fixedRotationAxis ),
        COPY_FROM( *prop_, iterLimit )
        COPY_FROM( *prop_, badIterStopCount )
        COPY_FROM( *prop_, exitVal )
        COPY_FROM( *prop_, mutualClosest )
    };

    icp.setParams( prop );
}

void mrICPSamplePoints( MRICP* icp_, float samplingVoxelSize )
{
    ARG( icp );
    icp.samplePoints( samplingVoxelSize );
}

MRAffineXf3f mrICPAutoSelectFloatXf( MRICP* icp_ )
{
    ARG( icp );
    RETURN( icp.autoSelectFloatXf() );
}

void mrICPUpdatePointPairs( MRICP* icp_ )
{
    ARG( icp );
    icp.updatePointPairs();
}

MRString* mrICPGetStatusInfo( const MRICP* icp_ )
{
    ARG( icp );
    RETURN_NEW( icp.getStatusInfo() );
}

size_t mrICPGetNumSamples( const MRICP* icp_ )
{
    ARG( icp );
    return icp.getNumSamples();
}

size_t mrICPGetNumActivePairs( const MRICP* icp_ )
{
    ARG( icp );
    return icp.getNumActivePairs();
}

float mrICPGetMeanSqDistToPoint( const MRICP* icp_ )
{
    ARG( icp );
    return icp.getMeanSqDistToPoint();
}

float mrICPGetMeanSqDistToPlane( const MRICP* icp_ )
{
    ARG( icp );
    return icp.getMeanSqDistToPlane();
}

const MRIPointPairs* mrICPGetFlt2RefPairs( const MRICP* icp_ )
{
    ARG( icp );
    return cast_to<MRIPointPairs>( &icp.getFlt2RefPairs() );
}

const MRIPointPairs* mrICPGetRef2FltPairs( const MRICP* icp_ )
{
    ARG( icp );
    return cast_to<MRIPointPairs>( &icp.getRef2FltPairs() );
}

MRAffineXf3f mrICPCalculateTransformation( MRICP* icp_ )
{
    ARG( icp );
    RETURN( icp.calculateTransformation() );
}

void mrICPFree( MRICP* icp_ )
{
    ARG_PTR( icp );
    delete icp;
}
