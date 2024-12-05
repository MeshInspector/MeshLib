#pragma once
#include "MRVoxelsFwd.h"

MR_EXTERN_C_BEGIN

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

/// converts OpenVDB Grid into mesh using Dual Marching Cubes algorithm
MRMESHC_API MRMesh* mrGridToMesh( const MRFloatGrid* grid, const MRGridToMeshSettings* settings, MRString** errorStr );

MR_EXTERN_C_END
