#pragma once

#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRSignDetectionMode.h"

#include <MRMesh/config.h>

MR_EXTERN_C_BEGIN

typedef struct MROffsetParameters
{
    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize;
    /// Progress callback
    MRProgressCallback callBack;
    /// determines the method to compute distance sign
    MRSignDetectionMode signDetectionMode;
    // TODO: fwn
    /// use FunctionVolume for voxel grid representation:
    ///  - memory consumption is approx. (z / (2 * thread_count)) lesser
    ///  - computation is about 2-3 times slower
    /// used only by \ref mrMcOffsetMesh and \ref mrSharpOffsetMesh functions
    bool memoryEfficient;
} MROffsetParameters;

/// initializes a default instance
MRMESHC_API MROffsetParameters mrOffsetParametersNew( void );

/// computes size of a cubical voxel to get approximately given number of voxels during rasterization
 MRMESHC_API float mrSuggestVoxelSize( MRMeshPart mp, float approxNumVoxels );

#ifndef MRMESH_NO_OPENVDB
/// Offsets mesh by converting it to distance field in voxels using OpenVDB library,
/// signDetectionMode = Unsigned(from OpenVDB) | OpenVDB | HoleWindingRule,
/// and then converts back using OpenVDB library (dual marching cubes),
/// so result mesh is always closed
/// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, MRString** errorString );

/// Offsets mesh by converting it to voxels and back two times
/// only closed meshes allowed (only Offset mode)
/// typically offsetA and offsetB have distinct signs
/// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrDoubleOffsetMesh( MRMeshPart mp, float offsetA, float offsetB, const MROffsetParameters* params, MRString** errorString );
#endif

/// Offsets mesh by converting it to distance field in voxels (using OpenVDB library if SignDetectionMode::OpenVDB or our implementation otherwise)
/// and back using standard Marching Cubes, as opposed to Dual Marching Cubes in offsetMesh(...)
/// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrMcOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, MRString** errorString );

/// Constructs a shell around selected mesh region with the properties that every point on the shall must
///  1. be located not further than given distance from selected mesh part,
///  2. be located not closer to not-selected mesh part than to selected mesh part.
/// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrMcShellMeshRegion( const MRMesh* mesh, const MRFaceBitSet* region, float offset, const MROffsetParameters* params, MRString** errorString );

/// allows the user to select in the parameters which offset algorithm to call
typedef enum MRGeneralOffsetParametersMode
{
    /// create mesh using dual marching cubes from OpenVDB library
    MRGeneralOffsetParametersModeSmooth = 0,
    /// create mesh using standard marching cubes implemented in MeshLib
    MRGeneralOffsetParametersModeStandard,
    /// create mesh using standard marching cubes with additional sharpening implemented in MeshLib
    MRGeneralOffsetParametersModeSharpening
} MRGeneralOffsetParametersMode;

typedef struct MRGeneralOffsetParameters
{
    // TODO: outSharpEdges
    /// minimal surface deviation to introduce new vertex in a voxel, measured in voxelSize
    float minNewVertDev;
    /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes), measured in voxelSize
    float maxNewRank2VertDev;
    /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes), measured in voxelSize
    float maxNewRank3VertDev;
    float maxOldVertPosCorrection;
    /// correct positions of the input vertices using reference mesh by not more than this distance, measured in voxelSize;
    /// big correction can be wrong and result from self-intersections in the reference mesh
    MRGeneralOffsetParametersMode mode;
} MRGeneralOffsetParameters;

/// initializes a default instance
MRMESHC_API MRGeneralOffsetParameters mrGeneralOffsetParametersNew( void );

/// Offsets mesh by converting it to voxels and back
/// post process result using reference mesh to sharpen features
/// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrSharpOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

/// Offsets mesh by converting it to voxels and back using one of three modes specified in the parameters
/// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrGeneralOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

/// in case of positive offset, returns the mesh consisting of offset mesh merged with inversed original mesh (thickening mode);
/// in case of negative offset, returns the mesh consisting of inversed offset mesh merged with original mesh (hollowing mode);
/// if your input mesh is open then please specify params.signDetectionMode = SignDetectionMode::Unsigned, and you will get open mesh (with several components) on output
/// if your input mesh is closed then please specify another sign detection mode, and you will get closed mesh (with several components) on output;
/// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrThickenMesh( const MRMesh* mesh, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

MR_EXTERN_C_END
