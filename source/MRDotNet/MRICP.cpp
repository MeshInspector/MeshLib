#include "MRICP.h"
#include "MRMeshOrPoints.h"
#include "MRVector3.h"
#include "MRBitSet.h"
#include "MRAffineXf.h"



MR_DOTNET_NAMESPACE_BEGIN

PointPairs::PointPairs( const MR::PointPairs& nativePairs )
{
    pairs = gcnew List<PointPair^>( int( nativePairs.vec.size() ) );
    for ( const auto& nativePair : nativePairs.vec )
    {
        PointPair^ pair = gcnew PointPair();
        pair->distSq = nativePair.distSq;
        pair->normalsAngleCos = nativePair.normalsAngleCos;
        pair->srcNorm = gcnew Vector3f( new MR::Vector3f( nativePair.srcNorm ) );
        pair->srcPoint = gcnew Vector3f( new MR::Vector3f( nativePair.srcPoint ) );
        pair->srcVertId = VertId( nativePair.srcVertId );

        pairs->Add( pair );
    }

    active = gcnew BitSet( new MR::BitSet( nativePairs.active ) );
}

MR::ICPProperties ICPProperties::ToNative()
{
    return MR::ICPProperties
    {
        .method = MR::ICPMethod( method ),
        .p2plAngleLimit = p2plAngleLimit,
        .p2plScaleLimit = p2plScaleLimit,
        .cosThreshold = cosThreshold,
        .distThresholdSq = distThresholdSq,
        .farDistFactor = farDistFactor,
        .icpMode = MR::ICPMode( icpMode ),
        .fixedRotationAxis = *fixedRotationAxis->vec(),
        .iterLimit = iterLimit,
        .badIterStopCount = badIterStopCount,
        .exitVal = exitVal,
        .mutualClosest = mutualClosest
    };
}

ICP::ICP( MeshOrPointsXf^ flt, MeshOrPointsXf^ ref, float samplingVoxelSize )
{
    if ( !flt )
        throw gcnew System::ArgumentNullException( "flt" );
    if ( !ref )
        throw gcnew System::ArgumentNullException( "ref" );

    icp_ = new MR::ICP( flt->ToNative(), ref->ToNative(), samplingVoxelSize );
}

ICP::ICP( MeshOrPointsXf^ flt, MeshOrPointsXf^ ref, BitSet^ fltSamples, BitSet^ refSamples )
{
    if ( !flt )
        throw gcnew System::ArgumentNullException( "flt" );
    if ( !ref )
        throw gcnew System::ArgumentNullException( "ref" );
    if ( !fltSamples )
        throw gcnew System::ArgumentNullException( "fltSamples" );
    if ( !refSamples )
        throw gcnew System::ArgumentNullException( "refSamples" );

    MR::VertBitSet nativeFltSamples( fltSamples->bitSet()->m_bits.begin(), fltSamples->bitSet()->m_bits.end() );
    MR::VertBitSet nativeRefSamples( refSamples->bitSet()->m_bits.begin(), refSamples->bitSet()->m_bits.end() );
    icp_ = new MR::ICP( flt->ToNative(), ref->ToNative(), nativeFltSamples, nativeRefSamples );
}

ICP::~ICP()
{
    delete icp_;
}

void ICP::SetParams( ICPProperties^ props )
{
    icp_->setParams( props->ToNative() );
}

void ICP::SamplePoints( float samplingVoxelSize )
{
    icp_->samplePoints( samplingVoxelSize );
}

void ICP::AutoSelectFloatXf()
{
    icp_->autoSelectFloatXf();
}

void ICP::UpdatePointPairs()
{
    icp_->updatePointPairs();
}

System::String^ ICP::GetStatusInfo()
{
    return gcnew System::String( icp_->getStatusInfo().c_str() );
}

int ICP::GetNumSamples()
{
    return int( icp_->getNumSamples() );
}

int ICP::GetNumActivePairs()
{
    return int( icp_->getNumActivePairs() );
}

float ICP::GetMeanSqDistToPlane()
{
    return icp_->getMeanSqDistToPlane();
}

float ICP::GetMeanSqDistToPoint()
{
    return icp_->getMeanSqDistToPoint();
}

PointPairs^ ICP::GetFlt2RefPairs()
{    
    return gcnew PointPairs( icp_->getFlt2RefPairs() );
}

PointPairs^ ICP::GetRef2FltPairs()
{    
    return gcnew PointPairs( icp_->getRef2FltPairs() );
}

AffineXf3f^ ICP::CalculateTransformation()
{
    return gcnew AffineXf3f( new MR::AffineXf3f( icp_->calculateTransformation() ) );
}

MR_DOTNET_NAMESPACE_END
