#include "MRICP.h"

#include "MRMesh/MRICP.h"

using namespace MR;

static_assert( sizeof( MRICPPairData ) == sizeof( MR::ICPPairData ) );

static_assert( sizeof( MRPointPair ) == sizeof( MR::PointPair ) );

#define COPY_FROM( obj, field ) . field = ( obj ). field ,

const MRICPPairData* mrIPointPairsGet( const MRIPointPairs* pp_, size_t idx )
{
    const auto& pp = *reinterpret_cast<const IPointPairs*>( pp_ );
    const auto& res = pp[idx];
    return reinterpret_cast<const MRICPPairData*>( &res );
}

MRICPPairData* mrIPointPairsGetRef( MRIPointPairs* pp_, size_t idx )
{
    auto& pp = *reinterpret_cast<IPointPairs*>( pp_ );
    auto& res = pp[idx];
    return reinterpret_cast<MRICPPairData*>( &res );
}

MRICPProperties mrICPPropertiesNew( void )
{
    static const ICPProperties def;
    return {
        .method = static_cast<MRICPMethod>( def.method ),
        COPY_FROM( def, p2plAngleLimit )
        COPY_FROM( def, p2plScaleLimit )
        COPY_FROM( def, cosThreshold )
        COPY_FROM( def, distThresholdSq )
        COPY_FROM( def, farDistFactor )
        .icpMode = static_cast<MRICPMode>( def.icpMode ),
        .fixedRotationAxis = reinterpret_cast<const MRVector3f&>( def.fixedRotationAxis ),
        COPY_FROM( def, iterLimit )
        COPY_FROM( def, badIterStopCount )
        COPY_FROM( def, exitVal )
        COPY_FROM( def, mutualClosest )
    };
}

MRICP* mrICPNew( const MRMeshOrPointsXf* flt_, const MRMeshOrPointsXf* ref_, float samplingVoxelSize )
{
    const auto& flt = *reinterpret_cast<const MeshOrPointsXf*>( flt_ );
    const auto& ref = *reinterpret_cast<const MeshOrPointsXf*>( ref_ );

    return reinterpret_cast<MRICP*>( new ICP( flt, ref, samplingVoxelSize ) );
}

void mrICPSetParams( MRICP* icp_, const MRICPProperties* prop_ )
{
    auto& icp = *reinterpret_cast<ICP*>( icp_ );

    const ICPProperties prop {
        .method = static_cast<ICPMethod>( prop_->method ),
        COPY_FROM( *prop_, p2plAngleLimit )
        COPY_FROM( *prop_, p2plScaleLimit )
        COPY_FROM( *prop_, cosThreshold )
        COPY_FROM( *prop_, distThresholdSq )
        COPY_FROM( *prop_, farDistFactor )
        .icpMode = static_cast<ICPMode>( prop_->icpMode ),
        .fixedRotationAxis = reinterpret_cast<const Vector3f&>( prop_->fixedRotationAxis ),
        COPY_FROM( *prop_, iterLimit )
        COPY_FROM( *prop_, badIterStopCount )
        COPY_FROM( *prop_, exitVal )
        COPY_FROM( *prop_, mutualClosest )
    };

    icp.setParams( prop );
}

void mrICPSamplePoints( MRICP* icp_, float samplingVoxelSize )
{
    auto& icp = *reinterpret_cast<ICP*>( icp_ );

    icp.samplePoints( samplingVoxelSize );
}

MRAffineXf3f mrICPAutoSelectFloatXf( MRICP* icp_ )
{
    auto& icp = *reinterpret_cast<ICP*>( icp_ );

    const auto res = icp.autoSelectFloatXf();
    return reinterpret_cast<const MRAffineXf3f&>( res );
}

void mrICPUpdatePointPairs( MRICP* icp_ )
{
    auto& icp = *reinterpret_cast<ICP*>( icp_ );

    icp.updatePointPairs();
}

MRString* mrICPGetStatusInfo( const MRICP* icp_ )
{
    const auto& icp = *reinterpret_cast<const ICP*>( icp_ );

    return reinterpret_cast<MRString*>( new std::string( icp.getStatusInfo() ) );
}

size_t mrICPGetNumSamples( const MRICP* icp_ )
{
    const auto& icp = *reinterpret_cast<const ICP*>( icp_ );

    return icp.getNumSamples();
}

size_t mrICPGetNumActivePairs( const MRICP* icp_ )
{
    const auto& icp = *reinterpret_cast<const ICP*>( icp_ );

    return icp.getNumActivePairs();
}

float mrICPGetMeanSqDistToPoint( const MRICP* icp_ )
{
    const auto& icp = *reinterpret_cast<const ICP*>( icp_ );

    return icp.getMeanSqDistToPoint();
}

float mrICPGetMeanSqDistToPlane( const MRICP* icp_ )
{
    const auto& icp = *reinterpret_cast<const ICP*>( icp_ );

    return icp.getMeanSqDistToPlane();
}

const MRIPointPairs* mrICPGetFlt2RefPairs( const MRICP* icp_ )
{
    const auto& icp = *reinterpret_cast<const ICP*>( icp_ );

    return reinterpret_cast<const MRIPointPairs*>( &icp.getFlt2RefPairs() );
}

const MRIPointPairs* mrICPGetRef2FltPairs( const MRICP* icp_ )
{
    const auto& icp = *reinterpret_cast<const ICP*>( icp_ );

    return reinterpret_cast<const MRIPointPairs*>( &icp.getRef2FltPairs() );
}

MRAffineXf3f mrICPCalculateTransformation( MRICP* icp_ )
{
    auto& icp = *reinterpret_cast<ICP*>( icp_ );

    const auto res = icp.calculateTransformation();
    return reinterpret_cast<const MRAffineXf3f&>( res );
}

void mrICPFree( MRICP* icp )
{
    delete reinterpret_cast<ICP*>( icp );
}
