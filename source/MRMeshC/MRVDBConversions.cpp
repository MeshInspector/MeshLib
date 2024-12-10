#include "MRVDBConversions.h"
#include "MRMesh.h"
#include "detail/TypeCast.h"
#include "MRAffineXf.h"

#include "MRMesh/MRMesh.h"
#include "MRVoxels/MRVDBConversions.h"
#include "MRVoxels/MRVoxelsVolume.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( FloatGrid )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST( Vector3i )
REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST2( std::string, MRString )

MRMeshToVolumeSettings mrVdbConversionsMeshToVolumeSettingsNew( void )
{
    static MeshToVolumeParams settings;    
    return MRMeshToVolumeSettings
    {
        .type = MRMeshToVolumeSettingsType( settings.type ),
        .surfaceOffset = settings.surfaceOffset,
        .voxelSize = auto_cast( settings.voxelSize ),
        .worldXf = mrAffineXf3fNew(),
        .outXf = nullptr,
        .cb = nullptr
    };
}

MRGridToMeshSettings mrVdbConversionsGridToMeshSettingsNew( void )
{
    static GridToMeshSettings settings;
    return MRGridToMeshSettings
    {
        .voxelSize = auto_cast( settings.voxelSize ),
        .isoValue = settings.isoValue,
        .adaptivity = settings.adaptivity,
        .maxFaces = settings.maxFaces,
        .maxVertices = settings.maxVertices,
        .relaxDisorientedTriangles = settings.relaxDisorientedTriangles,
        .cb = nullptr
    };
}

void mrVdbConversionsEvalGridMinMax( const MRFloatGrid* grid_, float* min, float* max )
{
    ARG(grid);
    evalGridMinMax( grid, *min, *max );
}

MRVdbVolume mrVdbConversionsMeshToVolume( const MRMesh* mesh_, const MRMeshToVolumeSettings* settings_, MRString** errorStr )
{
    ARG( mesh );

    MeshToVolumeParams settings
    {
        .type = MR::MeshToVolumeParams::Type( settings_->type ),
        .surfaceOffset = settings_->surfaceOffset,
        .voxelSize = auto_cast( settings_->voxelSize ),
        .worldXf = auto_cast( settings_->worldXf ),
        .outXf = (AffineXf3f*)settings_->outXf,
        .cb = settings_->cb
    };

    if ( auto resOrErr = meshToVolume( mesh, settings ) )
    {
        return MRVdbVolume
        {
            .data = auto_cast( new_from( std::move( resOrErr->data ) ) ),
            .dims = auto_cast( resOrErr->dims ),
            .voxelSize = auto_cast( resOrErr->voxelSize ),
            .min = resOrErr->min,
            .max = resOrErr->max
        };

    }
    else if ( errorStr )
    {
        *errorStr = auto_cast( new_from( std::move( resOrErr.error() ) ) );
    }

    return {};
}

MRVdbVolume mrVdbConversionsFloatGridToVdbVolume( const MRFloatGrid* grid_ )
{
    ARG( grid );
    auto res = floatGridToVdbVolume( grid );

    return MRVdbVolume
    {
        .data = auto_cast( new_from( std::move( res.data ) ) ),
        .dims = auto_cast( res.dims ),
        .voxelSize = auto_cast( res.voxelSize ),
        .min = res.min,
        .max = res.max
    };
}

MRMesh* mrVdbConversionsGridToMesh( const MRFloatGrid* grid_, const MRGridToMeshSettings* settings_, MRString** errorStr )
{
    ARG( grid );
    GridToMeshSettings settings
    {
        .voxelSize = auto_cast( settings_->voxelSize ),
        .isoValue = settings_->isoValue,
        .adaptivity = settings_->adaptivity,
        .maxFaces = settings_->maxFaces,
        .maxVertices = settings_->maxVertices,
        .cb = settings_->cb
    };
    auto res = gridToMesh( grid, settings );
    if ( res )
    {
        RETURN_NEW( *res );
    }

    if ( errorStr && !res )
        *errorStr = auto_cast( new_from( std::move( res.error() ) ) );

    return nullptr;
}
