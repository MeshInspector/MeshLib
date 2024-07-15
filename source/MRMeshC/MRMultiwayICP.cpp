#include "MRMultiwayICP.h"

#include "MRMesh/MRMultiwayICP.h"

#include <span>

using namespace MR;

#define COPY_FROM( obj, field ) . field = ( obj ). field ,

MRMultiwayICPSamplingParameters mrMultiwayIcpSamplingParametersNew( void )
{
    static const MultiwayICPSamplingParameters def;
    return {
        COPY_FROM( def, samplingVoxelSize )
        COPY_FROM( def, maxGroupSize )
        .cascadeMode = static_cast<MRMultiwayICPSamplingParametersCascadeMode>( def.cascadeMode ),
        .cb = nullptr,
    };
}

MRMultiwayICP* mrMultiwayICPNew( const MRMeshOrPointsXf* objects_, size_t objectsNum, const MRMultiwayICPSamplingParameters* samplingParams_ )
{
    std::span objects { reinterpret_cast<const MeshOrPointsXf*>( objects_ ), objectsNum };

    // TODO: cast instead of copying
    ICPObjects objectsVec( objects.begin(), objects.end() );

    MultiwayICPSamplingParameters samplingParams;
    if ( samplingParams_ )
    {
        const auto& src = *samplingParams_;
        samplingParams = {
            COPY_FROM( src, samplingVoxelSize )
            COPY_FROM( src, maxGroupSize )
            .cascadeMode = static_cast<MultiwayICPSamplingParameters::CascadeMode>( src.cascadeMode ),
            COPY_FROM( src, cb )
        };
    }

    return reinterpret_cast<MRMultiwayICP*>( new MultiwayICP( objectsVec, samplingParams ) );
}

MRVectorAffineXf3f* mrMultiwayICPCalculateTransformations( MRMultiwayICP* mwicp_, MRProgressCallback cb )
{
    auto& mwicp = *reinterpret_cast<MultiwayICP*>( mwicp_ );

    auto res = mwicp.calculateTransformations( cb );
    return reinterpret_cast<MRVectorAffineXf3f*>( new std::vector<AffineXf3f>( std::move( res.vec_ ) ) );
}

bool mrMultiwayICPResamplePoints( MRMultiwayICP* mwicp_, const MRMultiwayICPSamplingParameters* samplingParams_ )
{
    auto& mwicp = *reinterpret_cast<MultiwayICP*>( mwicp_ );

    MultiwayICPSamplingParameters samplingParams;
    if ( samplingParams_ )
    {
        const auto& src = *samplingParams_;
        samplingParams = {
            COPY_FROM( src, samplingVoxelSize )
            COPY_FROM( src, maxGroupSize )
            .cascadeMode = static_cast<MultiwayICPSamplingParameters::CascadeMode>( src.cascadeMode ),
            COPY_FROM( src, cb )
        };
    }

    return mwicp.resamplePoints( samplingParams );
}

bool mrMultiwayICPUpdateAllPointPairs( MRMultiwayICP* mwicp_, MRProgressCallback cb )
{
    auto& mwicp = *reinterpret_cast<MultiwayICP*>( mwicp_ );

    return mwicp.updateAllPointPairs( cb );
}

void mrMultiwayICPSetParams( MRMultiwayICP* mwicp_, const MRICPProperties* prop_ )
{
    auto& mwicp = *reinterpret_cast<MultiwayICP*>( mwicp_ );

    const ICPProperties prop {
        .method = static_cast<ICPMethod>( prop_->method ),
        COPY_FROM( *prop_, p2plAngleLimit )
        COPY_FROM( *prop_, p2plScaleLimit )
        COPY_FROM( *prop_, cosTreshold )
        COPY_FROM( *prop_, distThresholdSq )
        COPY_FROM( *prop_, farDistFactor )
        .icpMode = static_cast<ICPMode>( prop_->icpMode ),
        .fixedRotationAxis = reinterpret_cast<const Vector3f&>( prop_->fixedRotationAxis ),
        COPY_FROM( *prop_, iterLimit )
        COPY_FROM( *prop_, badIterStopCount )
        COPY_FROM( *prop_, exitVal )
        COPY_FROM( *prop_, mutualClosest )
    };

    mwicp.setParams( prop );
}
