#pragma once
#include "MRVoxelsFwd.h"
#include "MRAffineXf.h"

MR_EXTERN_C_BEGIN

// Conversion type
typedef enum MRMeshToVolumeSettingsType
{
    MRMeshToVolumeSettingsTypeSigned, // only closed meshes can be converted with signed type
    MRMeshToVolumeSettingsTypeUnsigned // this type leads to shell like iso-surfaces
} MRMeshToVolumeSettingsType;

// Parameters structure for meshToVolume function
typedef struct MRMeshToVolumeSettings
{
    MRMeshToVolumeSettingsType type; // Conversion type
    float surfaceOffset; // the number of voxels around surface to calculate distance in (should be positive)
    MRVector3f voxelSize;
    MRAffineXf3f worldXf; // mesh initial transform
    MRAffineXf3f* outXf; // optional output: xf to original mesh (respecting worldXf)
    MRProgressCallback cb;
} MRMeshToVolumeSettings;

MRMESHC_API MRMeshToVolumeSettings mrVdbConversionsMeshToVolumeSettingsNew( void );

// eval min max value from FloatGrid
MRMESHC_API void mrVdbConversionsEvalGridMinMax( const MRFloatGrid* grid, float* min, float* max );

// convert mesh to volume in (0,0,0)-(dim.x,dim.y,dim.z) grid box
MRMESHC_API MRVdbVolume mrVdbConversionsMeshToVolume( const MRMesh* mesh, const MRMeshToVolumeSettings* settings, MRString** errorStr );

// fills VdbVolume data from FloatGrid (does not fill voxels size, cause we expect it outside)
MRMESHC_API MRVdbVolume mrVdbConversionsFloatGridToVdbVolume( const MRFloatGrid* grid );

/// parameters of OpenVDB Grid to Mesh conversion using Dual Marching Cubes algorithm
typedef struct MRGridToMeshSettings
{
    /// the size of each voxel in the grid
    MRVector3f voxelSize;
    /// layer of grid with this value would be converted in mesh; isoValue can be negative only in level set grids
    float isoValue;
    /// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones (curvature can be lost on high values)
    float adaptivity;
    /// if the mesh exceeds this number of faces, an error returns
    int maxFaces;
    /// if the mesh exceeds this number of vertices, an error returns
    int maxVertices;
    bool relaxDisorientedTriangles;
    /// to receive progress and request cancellation
    MRProgressCallback cb;
} MRGridToMeshSettings;

MRMESHC_API MRGridToMeshSettings mrVdbConversionsGridToMeshSettingsNew( void );

/// converts OpenVDB Grid into mesh using Dual Marching Cubes algorithm
MRMESHC_API MRMesh* mrVdbConversionsGridToMesh( const MRFloatGrid* grid, const MRGridToMeshSettings* settings, MRString** errorStr );

MR_EXTERN_C_END
