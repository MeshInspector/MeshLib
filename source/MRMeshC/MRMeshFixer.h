#pragma once

#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRId.h"
MR_EXTERN_C_BEGIN

typedef struct MRMultipleEdge
{
    MRVertId v0;
    MRVertId v1;
} MRMultipleEdge;

/// returns all faces that complicate one of mesh holes;
/// hole is complicated if it passes via one vertex more than once;
/// deleting such faces simplifies the holes and makes them easier to fill
MRMESHC_API MRFaceBitSet* mrFindHoleComplicatingFaces( MRMesh* mesh );
/// finds faces having aspect ratio >= criticalAspectRatio
MRMESHC_API MRFaceBitSet* mrFindDegenerateFaces( const MRMeshPart* mp, float criticalAspectRatio, MRProgressCallback cb, MRString** errorString );
/// finds edges having length <= criticalLength
MRMESHC_API MRUndirectedEdgeBitSet* mrFindShortEdges( const MRMeshPart* mp, float criticalLength, MRProgressCallback cb, MRString** errorString );

/// resolves given multiple edges, but splitting all but one edge in each group
MRMESHC_API void fixMultipleEdges( MRMesh* mesh, const MRMultipleEdge* multipleEdges, size_t multipleEdgesNum );
/// finds and resolves multiple edges
MRMESHC_API void findAndFixMultipleEdges( MRMesh* mesh );

typedef enum MRFixMeshDegeneraciesParamsMode
{
    MRFixMeshDegeneraciesParamsModeDecimate, ///< use decimation only to fix degeneracies
    MRFixMeshDegeneraciesParamsModeRemesh,   ///< if decimation does not succeed, perform subdivision too
    MRFixMeshDegeneraciesParamsModeRemeshPatch ///< if both decimation and subdivision does not succeed, removes degenerate areas and fills occurred holes
} MRFixMeshDegeneraciesParamsMode;

typedef struct MRFixMeshDegeneraciesParams
{
    /// maximum permitted deviation from the original surface
    float maxDeviation;

    /// edges not longer than this value will be collapsed ignoring normals and aspect ratio checks
    float tinyEdgeLength;

    /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
    /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
    float criticalTriAspectRatio;

    /// Permit edge flips if it does not change dihedral angle more than on this value
    float maxAngleChange;

    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer;

    /// degenerations will be fixed only in given region, it is updated during the operation
    MRFaceBitSet* region;

    MRFixMeshDegeneraciesParamsMode mode;

    MRProgressCallback cb;
} MRFixMeshDegeneraciesParams;

MRMESHC_API MRFixMeshDegeneraciesParams mrFixMeshDegeneraciesParamsNew( void );

/// Fixes degenerate faces and short edges in mesh (changes topology)
MRMESHC_API void mrFixMeshDegeneracies( MRMesh* mesh, const MRFixMeshDegeneraciesParams* params, MRString** errorString );

MR_EXTERN_C_END
