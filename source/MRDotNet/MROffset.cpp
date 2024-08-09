#include "MROffset.h"
#include "MRMesh.h"
#include "MRBitSet.h"

#pragma managed( push, off )
#include <MRMesh/MROffset.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBitSet.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

float Offset::SuggestVoxelSize( MeshPart mp, float approxNumVoxels )
{
    if ( !mp.mesh )
        throw gcnew System::ArgumentNullException( "mesh" );

    MR::FaceBitSet nativeRegion;

    if ( mp.region )
        nativeRegion = MR::FaceBitSet( mp.region->bitSet()->m_bits.begin(), mp.region->bitSet()->m_bits.end() );

    MR::MeshPart nativeMp( *mp.mesh->getMesh(), mp.region ? &nativeRegion : nullptr );
    return MR::suggestVoxelSize( nativeMp, approxNumVoxels );
}

#ifndef MRMESH_NO_OPENVDB

Mesh^ Offset::OffsetMesh( MeshPart mp, float offset, OffsetParameters^ parameters )
{
    if ( !mp.mesh )
        throw gcnew System::ArgumentNullException( "mesh" );

    if ( !parameters )
        throw gcnew System::ArgumentNullException( "parameters" );

    if ( parameters->voxelSize <= 0.0f )
        throw gcnew System::ArgumentException( "voxelSize must be positive" );

    MR::OffsetParameters nativeParams;
    nativeParams.voxelSize = parameters->voxelSize;
    nativeParams.memoryEfficient = parameters->memoryEfficient;
    nativeParams.signDetectionMode = MR::SignDetectionMode( parameters->signDetectionMode );

    MR::FaceBitSet nativeRegion;
    
    if ( mp.region )
        nativeRegion = MR::FaceBitSet( mp.region->bitSet()->m_bits.begin(), mp.region->bitSet()->m_bits.end() );

    MR::MeshPart nativeMp( *mp.mesh->getMesh(), mp.region ? &nativeRegion : nullptr );
    auto meshOrErr = MR::offsetMesh( nativeMp, offset, nativeParams );

    if ( !meshOrErr )
        throw gcnew System::SystemException( gcnew System::String( meshOrErr.error().c_str() ) );

    return gcnew Mesh( new MR::Mesh( std::move( *meshOrErr ) ) );
}

Mesh^ Offset::DoubleOffsetMesh( MeshPart mp, float offsetA, float offsetB, OffsetParameters^ parameters )
{
    if ( !mp.mesh )
        throw gcnew System::ArgumentNullException( "mesh" );

    if ( !parameters )
        throw gcnew System::ArgumentNullException( "parameters" );

    if ( parameters->voxelSize <= 0.0f )
        throw gcnew System::ArgumentException( "voxelSize must be positive" );

    MR::OffsetParameters nativeParams;
    nativeParams.voxelSize = parameters->voxelSize;
    nativeParams.memoryEfficient = parameters->memoryEfficient;
    nativeParams.signDetectionMode = MR::SignDetectionMode( parameters->signDetectionMode );

    MR::FaceBitSet nativeRegion;

    if ( mp.region )
        nativeRegion = MR::FaceBitSet( mp.region->bitSet()->m_bits.begin(), mp.region->bitSet()->m_bits.end() );

    MR::MeshPart nativeMp( *mp.mesh->getMesh(), mp.region ? &nativeRegion : nullptr );
    auto meshOrErr = MR::doubleOffsetMesh( nativeMp, offsetA, offsetB, nativeParams );

    if ( !meshOrErr )
        throw gcnew System::SystemException( gcnew System::String( meshOrErr.error().c_str() ) );

    return gcnew Mesh( new MR::Mesh( std::move( *meshOrErr ) ) );
}
#endif

Mesh^ Offset::McOffsetMesh( MeshPart mp, float offset, OffsetParameters^ parameters )
{
    if ( !mp.mesh )
        throw gcnew System::ArgumentNullException( "mesh" );

    if ( !parameters )
        throw gcnew System::ArgumentNullException( "parameters" );

    if ( parameters->voxelSize <= 0.0f )
        throw gcnew System::ArgumentException( "voxelSize must be positive" );

    MR::OffsetParameters nativeParams;
    nativeParams.voxelSize = parameters->voxelSize;
    nativeParams.memoryEfficient = parameters->memoryEfficient;
    nativeParams.signDetectionMode = MR::SignDetectionMode( parameters->signDetectionMode );

    MR::FaceBitSet nativeRegion;
    if ( mp.region )
        nativeRegion = MR::FaceBitSet( mp.region->bitSet()->m_bits.begin(), mp.region->bitSet()->m_bits.end() );

    MR::MeshPart nativeMp( *mp.mesh->getMesh(), mp.region ? &nativeRegion : nullptr );
    auto meshOrErr = MR::mcOffsetMesh( nativeMp, offset, nativeParams );

    if ( !meshOrErr )
        throw gcnew System::SystemException( gcnew System::String( meshOrErr.error().c_str() ) );

    return gcnew Mesh( new MR::Mesh( std::move( *meshOrErr ) ) );
}

Mesh^ Offset::McShellMeshRegion( MeshPart mp, float offset, float voxelSize )
{
    if ( !mp.mesh )
        throw gcnew System::ArgumentNullException( "mesh" );

    if ( !mp.region )
        throw gcnew System::ArgumentNullException( "region" );

    if ( voxelSize <= 0.0f )
        throw gcnew System::ArgumentException( "voxelSize must be positive" );

    MR::FaceBitSet nativeRegion( mp.region->bitSet()->m_bits.begin(), mp.region->bitSet()->m_bits.end() );

    auto meshOrErr = mcShellMeshRegion( *mp.mesh->getMesh(), nativeRegion, offset, { .voxelSize = voxelSize } );

    if ( !meshOrErr )
        throw gcnew System::SystemException( gcnew System::String( meshOrErr.error().c_str() ) );

    return gcnew Mesh( new MR::Mesh( std::move( *meshOrErr ) ) );
}

Mesh^ Offset::SharpOffsetMesh( MeshPart mp, float offset, SharpOffsetParameters^ parameters )
{
    if ( !mp.mesh )
        throw gcnew System::ArgumentNullException( "mesh" );

    if ( !parameters )
        throw gcnew System::ArgumentNullException( "parameters" );

    if ( parameters->voxelSize <= 0.0f )
        throw gcnew System::ArgumentException( "voxelSize must be positive" );

    MR::SharpOffsetParameters nativeParams;
    nativeParams.voxelSize = parameters->voxelSize;
    nativeParams.memoryEfficient = parameters->memoryEfficient;
    nativeParams.signDetectionMode = MR::SignDetectionMode( parameters->signDetectionMode );
    nativeParams.maxNewRank2VertDev = parameters->maxNewRank2VertDev;
    nativeParams.maxNewRank3VertDev = parameters->maxNewRank3VertDev;
    nativeParams.maxOldVertPosCorrection = parameters->maxOldVertPosCorrection;
    nativeParams.minNewVertDev = parameters->minNewVertDev;

    MR::FaceBitSet nativeRegion;
    if ( mp.region )
        nativeRegion = MR::FaceBitSet( mp.region->bitSet()->m_bits.begin(), mp.region->bitSet()->m_bits.end() );

    MR::MeshPart nativeMp( *mp.mesh->getMesh(), mp.region ? &nativeRegion : nullptr );
    auto meshOrErr = MR::sharpOffsetMesh( nativeMp, offset, nativeParams );

    if ( !meshOrErr )
        throw gcnew System::SystemException( gcnew System::String( meshOrErr.error().c_str() ) );

    return gcnew Mesh( new MR::Mesh( std::move( *meshOrErr ) ) );
}

Mesh^ Offset::GeneralOffsetMesh( MeshPart mp, float offset, GeneralOffsetParameters^ parameters )
{
    if ( !mp.mesh )
        throw gcnew System::ArgumentNullException( "mesh" );

    if ( !parameters )
        throw gcnew System::ArgumentNullException( "parameters" );

    if ( parameters->voxelSize <= 0.0f )
        throw gcnew System::ArgumentException( "voxelSize must be positive" );

    MR::GeneralOffsetParameters nativeParams;
    nativeParams.voxelSize = parameters->voxelSize;
    nativeParams.memoryEfficient = parameters->memoryEfficient;
    nativeParams.signDetectionMode = MR::SignDetectionMode( parameters->signDetectionMode );
    nativeParams.maxNewRank2VertDev = parameters->maxNewRank2VertDev;
    nativeParams.maxNewRank3VertDev = parameters->maxNewRank3VertDev;
    nativeParams.maxOldVertPosCorrection = parameters->maxOldVertPosCorrection;
    nativeParams.minNewVertDev = parameters->minNewVertDev;
    nativeParams.mode = MR::GeneralOffsetParameters::Mode( parameters->mode );

    MR::FaceBitSet nativeRegion;
    if ( mp.region )
        nativeRegion = MR::FaceBitSet( mp.region->bitSet()->m_bits.begin(), mp.region->bitSet()->m_bits.end() );

    MR::MeshPart nativeMp( *mp.mesh->getMesh(), mp.region ? &nativeRegion : nullptr );
    auto meshOrErr = MR::generalOffsetMesh( nativeMp, offset, nativeParams );

    if ( !meshOrErr )
        throw gcnew System::SystemException( gcnew System::String( meshOrErr.error().c_str() ) );

    return gcnew Mesh( new MR::Mesh( std::move( *meshOrErr ) ) );
}

Mesh^ Offset::ThickenMesh( Mesh^ mesh, float offset, GeneralOffsetParameters^ parameters )
{
    if ( !mesh )
        throw gcnew System::ArgumentNullException( "mesh" );

    if ( !parameters )
        throw gcnew System::ArgumentNullException( "parameters" );

    if ( parameters->voxelSize <= 0.0f )
        throw gcnew System::ArgumentException( "voxelSize must be positive" );

    MR::GeneralOffsetParameters nativeParams;
    nativeParams.voxelSize = parameters->voxelSize;
    nativeParams.memoryEfficient = parameters->memoryEfficient;
    nativeParams.signDetectionMode = MR::SignDetectionMode( parameters->signDetectionMode );
    nativeParams.maxNewRank2VertDev = parameters->maxNewRank2VertDev;
    nativeParams.maxNewRank3VertDev = parameters->maxNewRank3VertDev;
    nativeParams.maxOldVertPosCorrection = parameters->maxOldVertPosCorrection;
    nativeParams.minNewVertDev = parameters->minNewVertDev;
    nativeParams.mode = MR::GeneralOffsetParameters::Mode( parameters->mode );

    auto meshOrErr = MR::thickenMesh( *mesh->getMesh(), offset, nativeParams );
    if ( !meshOrErr )
        throw gcnew System::SystemException( gcnew System::String( meshOrErr.error().c_str() ) );

    return gcnew Mesh( new MR::Mesh( std::move( *meshOrErr ) ) );
}

MR_DOTNET_NAMESPACE_END