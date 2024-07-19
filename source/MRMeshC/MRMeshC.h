#pragma once

#include "MRMesh.h"
#include "MRMeshBoolean.h"
#include "MRMeshDecimate.h"
#include "MRMeshLoad.h"
#include "MRMeshSave.h"
#include "MRMultiwayICP.h"
#include "MROffset.h"
#include "MRPointCloud.h"
#include "MRPointsLoad.h"
#include "MRPointsSave.h"

MR_EXTERN_C_BEGIN

/// Makes new mesh - result of boolean operation on mesh `A` and mesh `B`
/// \param meshA Input mesh `A`
/// \param meshB Input mesh `B`
/// \param operation CSG operation to perform
MRMESHC_API MRBooleanResult mrBoolean( const MRMesh* meshA, const MRMesh* meshB, MRBooleanOperation operation, const MRBooleanParameters* params );

/// Collapse edges in mesh region according to the settings
MRMESHC_API MRDecimateResult mrDecimateMesh( MRMesh* mesh, const MRDecimateSettings* settings );

/// Resolves degenerate triangles in given mesh
/// This function performs decimation, so it can affect topology
/// \return true if the mesh has been changed
MRMESHC_API bool mrResolveMeshDegenerations( MRMesh* mesh, const MRResolveMeshDegenSettings* settings );

/// Offsets mesh by converting it to distance field in voxels using OpenVDB library,
/// signDetectionMode = Unsigned(from OpenVDB) | OpenVDB | HoleWindingRule,
/// and then converts back using OpenVDB library (dual marching cubes),
/// so result mesh is always closed
/// if an error has occurred and errorString is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, MRString** errorString );

/// This class allows you to register many objects having similar parts
/// and known initial approximations of orientations/locations using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
MRMESHC_API MRMultiwayICP* mrMultiwayICPNew( const MRMeshOrPointsXf* objects, size_t objectsNum, const MRMultiwayICPSamplingParameters* samplingParams );

/// runs ICP algorithm given input objects, transformations, and parameters;
/// \return adjusted transformations of all objects to reach registered state
MRMESHC_API MRVectorAffineXf3f* mrMultiwayICPCalculateTransformations( MRMultiwayICP* mwicp, MRProgressCallback cb );

/// tune algorithm params before run calculateTransformations()
MRMESHC_API void mrMultiwayICPSetParams( MRMultiwayICP* mwicp, const MRICPProperties* prop );

/// constructs a mesh from vertex coordinates and a set of triangles with given ids
MRMESHC_API MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

/// constructs a mesh from vertex coordinates and a set of triangles with given ids;
/// unlike simple \ref mrMeshFromTriangles it tries to resolve non-manifold vertices by creating duplicate vertices
MRMESHC_API MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

/// constructs a mesh from point triples;
/// \param duplicateNonManifoldVertices = false, all coinciding points are given the same VertId in the result;
/// \param duplicateNonManifoldVertices = true, it tries to avoid non-manifold vertices by creating duplicate vertices with same coordinates
MRMESHC_API MRMesh* mrMeshNewFromPointTriples( const MRTriangle3f* posTriangles, size_t posTrianglesNum, bool duplicateNonManifoldVertices );

/// detects the format from file extension and loads mesh from it
/// if an error has occurred and errorStr is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrMeshLoadFromAnySupportedFormat( const char* file, MRString** errorStr );

/// detects the format from file extension and saves mesh to it
/// if an error has occurred and errorStr is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API void mrMeshSaveToAnySupportedFormat( const MRMesh* mesh, const char* file, MRString** errorStr );

/// creates a new PointCloud object
MRMESHC_API MRPointCloud* mrPointCloudFromPoints( const MRVector3f* points, size_t pointsNum );

/// detects the format from file extension and loads points from it
MRMESHC_API MRPointCloud* mrPointsLoadFromAnySupportedFormat( const char* filename, MRString** errorString );

/// detects the format from file extension and save points to it
MRMESHC_API void mrPointsSaveToAnySupportedFormat( const MRPointCloud* pc, const char* file, MRString** errorString );

MR_EXTERN_C_END
