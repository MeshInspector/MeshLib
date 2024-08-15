#include "MRMultiwayICP.h"
#include "MRMeshOrPoints.h"
#include "MRAffineXf.h"

#pragma managed( push, off )
#include <MRMesh/MRMultiwayICP.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

MultiwayICP::MultiwayICP( List<MeshOrPointsXf>^ objs, MultiwayICPSamplingParameters^ samplingParams )
{
    if ( !objs )
        throw gcnew System::ArgumentNullException( "objs" );
    if ( !samplingParams )
        throw gcnew System::ArgumentNullException( "samplingParams" );

    MR::ICPObjects nativeObjs;
    nativeObjs.reserve( objs->Count );

    for each ( MeshOrPointsXf^ obj in objs )
        nativeObjs.push_back( obj->ToNative() );

    MR::MultiwayICPSamplingParameters nativeSamplingParams
    {
        .samplingVoxelSize = samplingParams->samplingVoxelSize,
        .maxGroupSize = samplingParams->maxGroupSize,
        .cascadeMode = MR::MultiwayICPSamplingParameters::CascadeMode( samplingParams->cascadeMode )
    };

    icp_ = new MR::MultiwayICP( nativeObjs, nativeSamplingParams );
}

MultiwayICP::~MultiwayICP()
{
    delete icp_;
}

List<AffineXf3f^>^ MultiwayICP::CalculateTransformations()
{
    auto nativeTransforms = icp_->calculateTransformations();
    List<AffineXf3f^>^ res = gcnew List<AffineXf3f^>( int( nativeTransforms.size() ) );

    for ( auto& nativeTransform : nativeTransforms )
        res->Add( gcnew AffineXf3f( new MR::AffineXf3f( std::move( nativeTransform ) ) ) );
    
    return res;
}

void MultiwayICP::ResamplePoints( MultiwayICPSamplingParameters^ samplingParams )
{
    MR::MultiwayICPSamplingParameters nativeSamplingParams
    {
        .samplingVoxelSize = samplingParams->samplingVoxelSize,
        .maxGroupSize = samplingParams->maxGroupSize,
        .cascadeMode = MR::MultiwayICPSamplingParameters::CascadeMode( samplingParams->cascadeMode )
    };

    icp_->resamplePoints( nativeSamplingParams );
}

bool MultiwayICP::UpdateAllPointPairs()
{
    return icp_->updateAllPointPairs();
}

void MultiwayICP::SetParams( ICPProperties^ props )
{
    icp_->setParams( props->ToNative() );
}

float MultiwayICP::GetMeanSqDistToPoint()
{
    return icp_->getMeanSqDistToPoint();
}

float MultiwayICP::GetMeanSqDistToPoint( double value )
{
    return icp_->getMeanSqDistToPoint( &value );
}

float MultiwayICP::GetMeanSqDistToPlane()
{
    return icp_->getMeanSqDistToPlane();
}

float MultiwayICP::GetMeanSqDistToPlane( double value )
{
    return icp_->getMeanSqDistToPlane( &value );
}

int MultiwayICP::GetNumSamples()
{
    return int( icp_->getNumSamples() );
}

int MultiwayICP::GetNumActivePairs()
{
    return int( icp_->getNumActivePairs() );
}

MR_DOTNET_NAMESPACE_END
