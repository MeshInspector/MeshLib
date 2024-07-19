#include "MROffset.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MROffset.h"

using namespace MR;

#define COPY_FROM( obj, field ) . field = ( obj ). field

MROffsetParameters mrOffsetParametersNew()
{
    static const OffsetParameters def;
    return {
        COPY_FROM( def, voxelSize ),
        .callBack = nullptr,
        .signDetectionMode = static_cast<MRSignDetectionMode>( def.signDetectionMode ),
        COPY_FROM( def, memoryEfficient ),
    };
}

float mrSuggestVoxelSize( MRMeshPart mp, float approxNumVoxels )
{
    return suggestVoxelSize(
        MeshPart {
            *reinterpret_cast<const Mesh*>( mp.mesh ),
            reinterpret_cast<const FaceBitSet*>( mp.region ),
        },
        approxNumVoxels
    );
}

#ifndef MRMESH_NO_OPENVDB
MRMesh* mrOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params_, MRString** errorString )
{
    OffsetParameters params;
    if ( params_ )
    {
        const auto& src = *params_;
        params COPY_FROM( src, voxelSize );
        params COPY_FROM( src, callBack );
        params.signDetectionMode = static_cast<SignDetectionMode>( src.signDetectionMode );
        params COPY_FROM( src, memoryEfficient );
    }

    auto res = offsetMesh(
        MeshPart {
            *reinterpret_cast<const Mesh*>( mp.mesh ),
            reinterpret_cast<const FaceBitSet*>( mp.region ),
        },
        offset,
        params
    );
    if ( res )
    {
        auto* resMesh = new Mesh( std::move( *res ) );
        return reinterpret_cast<MRMesh*>( resMesh );
    }
    if ( errorString != nullptr )
    {
        auto* str = new std::string( std::move( res.error() ) );
        *errorString = reinterpret_cast<MRString*>( str );
    }
    return nullptr;
}

MRMesh* mrDoubleOffsetMesh( MRMeshPart mp, float offsetA, float offsetB, const MROffsetParameters* params_, MRString** errorString )
{
    OffsetParameters params;
    if ( params_ )
    {
        const auto& src = *params_;
        params COPY_FROM( src, voxelSize );
        params COPY_FROM( src, callBack );
        params.signDetectionMode = static_cast<SignDetectionMode>( src.signDetectionMode );
        params COPY_FROM( src, memoryEfficient );
    }

    auto res = doubleOffsetMesh(
        MeshPart {
            *reinterpret_cast<const Mesh*>( mp.mesh ),
            reinterpret_cast<const FaceBitSet*>( mp.region ),
        },
        offsetA,
        offsetB,
        params
    );
    if ( res )
    {
        auto* resMesh = new Mesh( std::move( *res ) );
        return reinterpret_cast<MRMesh*>( resMesh );
    }
    if ( errorString != nullptr )
    {
        auto* str = new std::string( std::move( res.error() ) );
        *errorString = reinterpret_cast<MRString*>( str );
    }
    return nullptr;
}
#endif

MRMesh* mrMcOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params_, MRString** errorString )
{
    OffsetParameters params;
    if ( params_ )
    {
        const auto& src = *params_;
        params COPY_FROM( src, voxelSize );
        params COPY_FROM( src, callBack );
        params.signDetectionMode = static_cast<SignDetectionMode>( src.signDetectionMode );
        params COPY_FROM( src, memoryEfficient );
    }

    auto res = mcOffsetMesh(
        MeshPart {
            *reinterpret_cast<const Mesh*>( mp.mesh ),
            reinterpret_cast<const FaceBitSet*>( mp.region ),
        },
        offset,
        params
    );
    if ( res )
    {
        auto* resMesh = new Mesh( std::move( *res ) );
        return reinterpret_cast<MRMesh*>( resMesh );
    }
    if ( errorString != nullptr )
    {
        auto* str = new std::string( std::move( res.error() ) );
        *errorString = reinterpret_cast<MRString*>( str );
    }
    return nullptr;
}

MRMesh* mrMcShellMeshRegion( const MRMesh* mesh, const MRFaceBitSet* region, float offset, const MROffsetParameters* params_, MRString** errorString )
{
    OffsetParameters params;
    if ( params_ )
    {
        const auto& src = *params_;
        params COPY_FROM( src, voxelSize );
        params COPY_FROM( src, callBack );
        params.signDetectionMode = static_cast<SignDetectionMode>( src.signDetectionMode );
        params COPY_FROM( src, memoryEfficient );
    }

    auto res = mcShellMeshRegion(
        *reinterpret_cast<const Mesh*>( mesh ),
        *reinterpret_cast<const FaceBitSet*>( region ),
        offset,
        params
    );
    if ( res )
    {
        auto* resMesh = new Mesh( std::move( *res ) );
        return reinterpret_cast<MRMesh*>( resMesh );
    }
    if ( errorString != nullptr )
    {
        auto* str = new std::string( std::move( res.error() ) );
        *errorString = reinterpret_cast<MRString*>( str );
    }
    return nullptr;
}

MRGeneralOffsetParameters mrGeneralOffsetParametersNew()
{
    static const GeneralOffsetParameters def;
    return {
        // TODO: outSharpEdges
        COPY_FROM( def, minNewVertDev ),
        COPY_FROM( def, maxNewRank2VertDev ),
        COPY_FROM( def, maxNewRank3VertDev ),
        COPY_FROM( def, maxOldVertPosCorrection ),
    };
}

MRMesh* mrSharpOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params_, const MRGeneralOffsetParameters* generalParams_, MRString** errorString )
{
    GeneralOffsetParameters params;
    if ( params_ )
    {
        const auto& src = *params_;
        params COPY_FROM( src, voxelSize );
        params COPY_FROM( src, callBack );
        params.signDetectionMode = static_cast<SignDetectionMode>( src.signDetectionMode );
        params COPY_FROM( src, memoryEfficient );
    }
    if ( generalParams_ )
    {
        const auto& src = *generalParams_;
        params COPY_FROM( src, minNewVertDev );
        params COPY_FROM( src, maxNewRank2VertDev );
        params COPY_FROM( src, maxNewRank3VertDev );
        params COPY_FROM( src, maxOldVertPosCorrection );
    }

    auto res = sharpOffsetMesh(
        MeshPart {
            *reinterpret_cast<const Mesh*>( mp.mesh ),
            reinterpret_cast<const FaceBitSet*>( mp.region ),
        },
        offset,
        params
    );
    if ( res )
    {
        auto* resMesh = new Mesh( std::move( *res ) );
        return reinterpret_cast<MRMesh*>( resMesh );
    }
    if ( errorString != nullptr )
    {
        auto* str = new std::string( std::move( res.error() ) );
        *errorString = reinterpret_cast<MRString*>( str );
    }
    return nullptr;
}

MRMesh* mrGeneralOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params_, const MRGeneralOffsetParameters* generalParams_, MRString** errorString )
{
    GeneralOffsetParameters params;
    if ( params_ )
    {
        const auto& src = *params_;
        params COPY_FROM( src, voxelSize );
        params COPY_FROM( src, callBack );
        params.signDetectionMode = static_cast<SignDetectionMode>( src.signDetectionMode );
        params COPY_FROM( src, memoryEfficient );
    }
    if ( generalParams_ )
    {
        const auto& src = *generalParams_;
        params COPY_FROM( src, minNewVertDev );
        params COPY_FROM( src, maxNewRank2VertDev );
        params COPY_FROM( src, maxNewRank3VertDev );
        params COPY_FROM( src, maxOldVertPosCorrection );
    }

    auto res = generalOffsetMesh(
        MeshPart {
            *reinterpret_cast<const Mesh*>( mp.mesh ),
            reinterpret_cast<const FaceBitSet*>( mp.region ),
        },
        offset,
        params
    );
    if ( res )
    {
        auto* resMesh = new Mesh( std::move( *res ) );
        return reinterpret_cast<MRMesh*>( resMesh );
    }
    if ( errorString != nullptr )
    {
        auto* str = new std::string( std::move( res.error() ) );
        *errorString = reinterpret_cast<MRString*>( str );
    }
    return nullptr;
}

MRMesh* mrThickenMesh( const MRMesh* mesh, float offset, const MROffsetParameters* params_, const MRGeneralOffsetParameters* generalParams_, MRString** errorString )
{
    GeneralOffsetParameters params;
    if ( params_ )
    {
        const auto& src = *params_;
        params COPY_FROM( src, voxelSize );
        params COPY_FROM( src, callBack );
        params.signDetectionMode = static_cast<SignDetectionMode>( src.signDetectionMode );
        params COPY_FROM( src, memoryEfficient );
    }
    if ( generalParams_ )
    {
        const auto& src = *generalParams_;
        params COPY_FROM( src, minNewVertDev );
        params COPY_FROM( src, maxNewRank2VertDev );
        params COPY_FROM( src, maxNewRank3VertDev );
        params COPY_FROM( src, maxOldVertPosCorrection );
    }

    auto res = thickenMesh(
        *reinterpret_cast<const Mesh*>( mesh ),
        offset,
        params
    );
    if ( res )
    {
        auto* resMesh = new Mesh( std::move( *res ) );
        return reinterpret_cast<MRMesh*>( resMesh );
    }
    if ( errorString != nullptr )
    {
        auto* str = new std::string( std::move( res.error() ) );
        *errorString = reinterpret_cast<MRString*>( str );
    }
    return nullptr;
}
