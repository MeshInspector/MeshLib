#include "MROffset.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMesh.h"
#include "MRVoxels/MROffset.h"

using namespace MR;

REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( SignDetectionMode )
REGISTER_AUTO_CAST2( std::string, MRString )

#define COPY_FROM( obj, field ) . field = ( obj ). field

namespace
{

MeshPart cast( MRMeshPart mp )
{
    return {
        *auto_cast( mp.mesh ),
        auto_cast( mp.region )
    };
}

} // namespace

MROffsetParameters mrOffsetParametersNew()
{
    static const OffsetParameters def;
    return {
        COPY_FROM( def, voxelSize ),
        .callBack = nullptr,
        .signDetectionMode = auto_cast( def.signDetectionMode ),
        COPY_FROM( def, memoryEfficient ),
    };
}

float mrSuggestVoxelSize( MRMeshPart mp, float approxNumVoxels )
{
    return suggestVoxelSize( cast( mp ), approxNumVoxels );
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
        params.signDetectionMode = auto_cast( src.signDetectionMode );
        params COPY_FROM( src, memoryEfficient );
    }

    auto res = offsetMesh( cast( mp ), offset, params );
    if ( res )
    {
        RETURN_NEW( std::move( *res ) );
    }
    else
    {
        if ( errorString != nullptr )
            *errorString = auto_cast( new_from( std::move( res.error() ) ) );
        return nullptr;
    }
}

MRMesh* mrDoubleOffsetMesh( MRMeshPart mp, float offsetA, float offsetB, const MROffsetParameters* params_, MRString** errorString )
{
    OffsetParameters params;
    if ( params_ )
    {
        const auto& src = *params_;
        params COPY_FROM( src, voxelSize );
        params COPY_FROM( src, callBack );
        params.signDetectionMode = auto_cast( src.signDetectionMode );
        params COPY_FROM( src, memoryEfficient );
    }

    auto res = doubleOffsetMesh( cast( mp ), offsetA, offsetB, params );
    if ( res )
    {
        RETURN_NEW( std::move( *res ) );
    }
    else
    {
        if ( errorString != nullptr )
            *errorString = auto_cast( new_from( std::move( res.error() ) ) );
        return nullptr;
    }
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

    auto res = mcOffsetMesh( cast( mp ), offset, params );
    if ( res )
    {
        RETURN_NEW( std::move( *res ) );
    }
    else
    {
        if ( errorString != nullptr )
            *errorString = auto_cast( new_from( std::move( res.error() ) ) );
        return nullptr;
    }
}

MRMesh* mrMcShellMeshRegion( const MRMesh* mesh_, const MRFaceBitSet* region_, float offset, const MROffsetParameters* params_, MRString** errorString )
{
    ARG( mesh ); ARG( region );

    OffsetParameters params;
    if ( params_ )
    {
        const auto& src = *params_;
        params COPY_FROM( src, voxelSize );
        params COPY_FROM( src, callBack );
        params.signDetectionMode = auto_cast( src.signDetectionMode );
        params COPY_FROM( src, memoryEfficient );
    }

    auto res = mcShellMeshRegion( mesh, region, offset, params );
    if ( res )
    {
        RETURN_NEW( std::move( *res ) );
    }
    else
    {
        if ( errorString != nullptr )
            *errorString = auto_cast( new_from( std::move( res.error() ) ) );
        return nullptr;
    }
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

    auto res = sharpOffsetMesh( cast( mp ), offset, params );
    if ( res )
    {
        RETURN_NEW( std::move( *res ) );
    }
    else
    {
        if ( errorString != nullptr )
            *errorString = auto_cast( new_from( std::move( res.error() ) ) );
        return nullptr;
    }
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

    auto res = generalOffsetMesh( cast( mp ), offset, params );
    if ( res )
    {
        RETURN_NEW( std::move( *res ) );
    }
    else
    {
        if ( errorString != nullptr )
            *errorString = auto_cast( new_from( std::move( res.error() ) ) );
        return nullptr;
    }
}

MRMesh* mrThickenMesh( const MRMesh* mesh_, float offset, const MROffsetParameters* params_, const MRGeneralOffsetParameters* generalParams_, MRString** errorString )
{
    ARG( mesh );

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

    auto res = thickenMesh( mesh, offset, params );
    if ( res )
    {
        RETURN_NEW( std::move( *res ) );
    }
    else
    {
        if ( errorString != nullptr )
            *errorString = auto_cast( new_from( std::move( res.error() ) ) );
        return nullptr;
    }
}
