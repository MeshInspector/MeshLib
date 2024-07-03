#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef _WIN32
#   ifdef MRMESHC_EXPORT
#       define MRMESHC_API __declspec( dllexport )
#   else
#       define MRMESHC_API __declspec( dllimport )
#   endif
#else
#   define MRMESHC_API __attribute__( ( visibility( "default" ) ) )
#endif

#ifdef __cplusplus
#define MR_EXTERN_C_BEGIN extern "C" {
#define MR_EXTERN_C_END }
#else
#define MR_EXTERN_C_BEGIN
#define MR_EXTERN_C_END
#endif

MR_EXTERN_C_BEGIN

typedef struct MRString MRString;

typedef struct MRBitSet MRBitSet;
typedef MRBitSet MRFaceBitSet;
typedef MRBitSet MRVertBitSet;

typedef struct MRMeshTopology MRMeshTopology;
typedef struct MRMesh MRMesh;

typedef struct MRTriangulation MRTriangulation;

typedef struct MREdgePath MREdgePath;
typedef MREdgePath MREdgeLoop;

typedef bool (*MRProgressCallback)( float );

/// three-dimensional vector
typedef struct MRVector3f
{
    float x;
    float y;
    float z;
} MRVector3f;

/// a set of 3 vectors; useful for representing a face via its vertex coordinates
typedef MRVector3f MRTriangle3f[3];

/// arbitrary row-major 3x3 matrix
typedef struct MRMatrix3f
{
    MRVector3f x;
    MRVector3f y;
    MRVector3f z;
} MRMatrix3f;

/// affine transformation: y = A*x + b, where A in VxV, and b in V
typedef struct MRAffineXf3f
{
    MRMatrix3f A;
    MRVector3f b;
} MRAffineXf3f;

/// initializes a default instance
MRMESHC_API MRAffineXf3f mrAffineXf3fNew( void );

/// creates translation-only transformation (with identity linear component)
MRMESHC_API MRAffineXf3f mrAffineXf3fTranslation( const MRVector3f* b );

/// creates linear-only transformation (without translation)
MRMESHC_API MRAffineXf3f mrAffineXf3fLinear( const MRMatrix3f* A );

/// composition of two transformations:
/// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
MRMESHC_API MRAffineXf3f mrAffineXf3fMul( const MRAffineXf3f* a, const MRAffineXf3f* b );

/// gets read-only access to the underlying blocks of a bitset
MRMESHC_API const uint64_t* mrBitSetBlocks( const MRBitSet* bs );

/// gets count of the underlying blocks of a bitset
MRMESHC_API size_t mrBitSetBlocksNum( const MRBitSet* bs );

/// gets total length of a bitset
MRMESHC_API size_t mrBitSetSize( const MRBitSet* bs );

/// checks if two bitsets are equal (have the same length and identical bit values)
MRMESHC_API bool mrBitSetEq( const MRBitSet* a, const MRBitSet* b );

/// deallocates a BitSet object
MRMESHC_API void mrBitSetFree( MRBitSet* bs );

/// creates a copy of a FaceBitSet object
MRMESHC_API MRFaceBitSet* mrFaceBitSetCopy( const MRFaceBitSet* fbs );

/// deallocates a FaceBitSet object
MRMESHC_API void mrFaceBitSetFree( MRFaceBitSet* fbs );

/// Available CSG operations
typedef enum MRBooleanOperation
{
    /// Part of mesh `A` that is inside of mesh `B`
    MRBooleanOperationInsideA = 0,
    /// Part of mesh `B` that is inside of mesh `A`
    MRBooleanOperationInsideB,
    /// Part of mesh `A` that is outside of mesh `B`
    MRBooleanOperationOutsideA,
    /// Part of mesh `B` that is outside of mesh `A`
    MRBooleanOperationOutsideB,
    /// Union surface of two meshes (outside parts)
    MRBooleanOperationUnion,
    /// Intersection surface of two meshes (inside parts)
    MRBooleanOperationIntersection,
    /// Surface of mesh `B` - surface of mesh `A` (outside `B` - inside `A`)
    MRBooleanOperationDifferenceBA,
    /// Surface of mesh `A` - surface of mesh `B` (outside `A` - inside `B`)
    MRBooleanOperationDifferenceAB,
    /// not a valid operation
    MRBooleanOperationCount
} MRBooleanOperation;

/// creates a mesh representing a cube
/// base is the "lower" corner of the cube coordinates
MRMESHC_API MRMesh* mrMakeCube( const MRVector3f* size, const MRVector3f* base );

/// optional parameters for \ref mrMakeCylinderAdvanced
typedef struct MRMakeCylinderAdvancedParameters
{
    float radius0;
    float radius1;
    float startAngle;
    float arcSize;
    float length;
    int resolution;
} MRMakeCylinderAdvancedParameters;

/// initializes a default instance
MRMESHC_API MRMakeCylinderAdvancedParameters mrMakeCylinderAdvancedParametersNew( void );

// creates a mesh representing a cylinder
MRMESHC_API MRMesh* mrMakeCylinderAdvanced( const MRMakeCylinderAdvancedParameters* params );

/// edge index
typedef struct MREdgeId { int id; } MREdgeId;
/// face index
typedef struct MRFaceId { int id; } MRFaceId;
/// vertex index
typedef struct MRVertId { int id; } MRVertId;

/// a set of 3 vertices; useful for representing a face via its vertex indices
typedef MRVertId MRThreeVertIds[3];

/// initializes an identity matrix
MRMESHC_API MRMatrix3f mrMatrix3fIdentity( void );

/// creates a matrix representing rotation around given axis on given angle
MRMESHC_API MRMatrix3f mrMatrix3fRotationScalar( const MRVector3f* axis, float angle );

/// creates a matrix representing rotation that after application to (from) makes (to) vector
MRMESHC_API MRMatrix3f mrMatrix3fRotationVector( const MRVector3f* from, const MRVector3f* to );

/// multiplies two matrices
MRMESHC_API MRMatrix3f mrMatrix3fMul( const MRMatrix3f* a, const MRMatrix3f* b );

/// optional parameters for \ref mrBoolean
typedef struct MRBooleanParameters
{
    /// Transform from mesh `B` space to mesh `A` space
    const MRAffineXf3f* rigidB2A;
    // TODO: mapper
    // TODO: outPreCutA
    // TODO: outPreCutB
    // TODO: outCutEdges
    /// By default produce valid operation on disconnected components
    /// if set merge all non-intersecting components
    bool mergeAllNonIntersectingComponents;
    /// Progress callback
    MRProgressCallback cb;
} MRBooleanParameters;

/// This structure store result mesh of mrBoolean or some error info
typedef struct MRBooleanResult
{
    /// Result mesh of boolean operation, if error occurred it would be empty
    MRMesh* mesh;
    // TODO: meshABadContourFaces
    // TODO: meshBBadContourFaces
    /// Holds error message, empty if boolean succeed
    MRString* errorString;
} MRBooleanResult;

/// Makes new mesh - result of boolean operation on mesh `A` and mesh `B`
/// \param meshA Input mesh `A`
/// \param meshB Input mesh `B`
/// \param operation CSG operation to perform
MRMESHC_API MRBooleanResult mrBoolean( const MRMesh* meshA, const MRMesh* meshB, MRBooleanOperation operation, const MRBooleanParameters* params );

/// Defines the order of edge collapses inside Decimate algorithm
typedef enum MRDecimateStrategy
{
    /// the next edge to collapse will be the one that introduced minimal error to the surface
    MRDecimateStrategyMinimizeError = 0,
    /// the next edge to collapse will be the shortest one
    MRDecimateStrategyShortestEdgeFirst
} MRDecimateStrategy;

/// parameters for \ref mrDecimateMesh
typedef struct MRDecimateSettings
{
    MRDecimateStrategy strategy;
    /// for DecimateStrategy::MinimizeError:
    ///   stop the decimation as soon as the estimated distance deviation from the original mesh is more than this value
    /// for DecimateStrategy::ShortestEdgeFirst only:
    ///   stop the decimation as soon as the shortest edge in the mesh is greater than this value
    float maxError;
    /// Maximal possible edge length created during decimation
    float maxEdgeLen;
    /// Maximal shift of a boundary during one edge collapse
    float maxBdShift;
    /// Maximal possible aspect ratio of a triangle introduced during decimation
    float maxTriangleAspectRatio;
    /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
    /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
    float criticalTriAspectRatio;
    /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
    float tinyEdgeLength;
    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer;
    /// if true then after each edge collapse the position of remaining vertex is optimized to
    /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
    bool optimizeVertexPos;
    /// Limit on the number of deleted vertices
    int maxDeletedVertices;
    /// Limit on the number of deleted faces
    int maxDeletedFaces;
    /// Region on mesh to be decimated, it is updated during the operation
    MRFaceBitSet* region;
    // TODO: notFlippable
    /// Whether to allow collapse of edges incident to notFlippable edges,
    /// which can move vertices of notFlippable edges unless they are fixed
    bool collapseNearNotFlippable;
    // TODO: edgesToCollapse
    // TODO: twinMap
    /// Whether to allow collapsing or flipping edges having at least one vertex on (region) boundary
    bool touchNearBdEdges;
    /// touchBdVerts=true: allow moving and eliminating boundary vertices during edge collapses;
    /// touchBdVerts=false: allow only collapsing an edge having only one boundary vertex in that vertex, so position and count of boundary vertices do not change;
    /// this setting is ignored if touchNearBdEdges=false
    bool touchBdVerts;
    // TODO: bdVerts
    /// Permit edge flips (in addition to collapsing) to improve Delone quality of the mesh
    /// if it does not change dihedral angle more than on this value (negative value prohibits any edge flips)
    float maxAngleChange;
    // TODO: preCollapse
    // TODO: adjustCollapse
    // TODO: onEdgeDel
    // TODO: vertForms
    /// whether to pack mesh at the end
    bool packMesh;
    /// callback to report algorithm progress and cancel it by user request
    MRProgressCallback progressCallback;
    /// If this value is more than 1, then virtually subdivides the mesh on given number of parts to process them in parallel (using many threads);
    /// unlike \ref mrDecimateParallelMesh it does not create copies of mesh regions, so may take less memory to operate;
    /// IMPORTANT: please call mrMeshPackOptimally before calling decimating with subdivideParts > 1, otherwise performance will be bad
    int subdivideParts;
    /// After parallel decimation of all mesh parts is done, whether to perform final decimation of whole mesh region
    /// to eliminate small edges near the border of individual parts
    bool decimateBetweenParts;
    // TODO: partFaces
    /// minimum number of faces in one subdivision part for ( subdivideParts > 1 ) mode
    int minFacesInPart;
} MRDecimateSettings;

/// initializes a default instance
MRMESHC_API MRDecimateSettings mrDecimateSettingsNew( void );

/// results of mrDecimateMesh
typedef struct MRDecimateResult
{
    /// Number deleted verts. Same as the number of performed collapses
    int vertsDeleted;
    /// Number deleted faces
    int facesDeleted;
    /// for DecimateStrategy::MinimizeError:
    ///    estimated distance deviation of decimated mesh from the original mesh
    /// for DecimateStrategy::ShortestEdgeFirst:
    ///    the shortest remaining edge in the mesh
    float errorIntroduced;
    /// whether the algorithm was cancelled by the callback
    bool cancelled;
} MRDecimateResult;

/// Collapse edges in mesh region according to the settings
MRMESHC_API MRDecimateResult mrDecimateMesh( MRMesh* mesh, const MRDecimateSettings* settings );

/// parameters for \ref mrResolveMeshDegenerations
typedef struct MRResolveMeshDegenSettings
{
    /// maximum permitted deviation from the original surface
    float maxDeviation;
    /// edges not longer than this value will be collapsed ignoring normals and aspect ratio checks
    float tinyEdgeLength;
    /// Permit edge flips if it does not change dihedral angle more than on this value
    float maxAngleChange;
    /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
    /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
    float criticalAspectRatio;
    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer;
    /// degenerations will be fixed only in given region, which is updated during the processing
    MRFaceBitSet* region;
} MRResolveMeshDegenSettings;

/// initializes a default instance
MRMESHC_API MRResolveMeshDegenSettings mrResolveMeshDegenSettingsNew( void );

/// Resolves degenerate triangles in given mesh
/// This function performs decimation, so it can affect topology
/// \return true if the mesh has been changed
MRMESHC_API bool mrResolveMeshDegenerations( MRMesh* mesh, const MRResolveMeshDegenSettings* settings );

/// parameters for \ref mrRemesh
typedef struct MRRemeshSettings
{
    /// the algorithm will try to keep the length of all edges close to this value,
    /// splitting the edges longer than targetEdgeLen, and then eliminating the edges shorter than targetEdgeLen
    float targetEdgeLen;
    /// Maximum number of edge splits allowed during subdivision
    int maxEdgeSplits;
    /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value
    float maxAngleChangeAfterFlip;
    /// Maximal shift of a boundary during one edge collapse
    float maxBdShift;
    /// This option in subdivision works best for natural surfaces, where all triangles are close to equilateral and have similar area,
    /// and no sharp edges in between
    bool useCurvature;
    /// the number of iterations of final relaxation of mesh vertices;
    /// few iterations can give almost perfect uniformity of the vertices and edge lengths but deviate from the original surface
    int finalRelaxIters;
    /// if true prevents the surface from shrinkage after many iterations
    bool finalRelaxNoShrinkage;
    /// Region on mesh to be changed, it is updated during the operation
    MRFaceBitSet* region;
    // TODO: notFlippable
    /// whether to pack mesh at the end
    bool packMesh;
    /// if true, then every new vertex after subdivision will be projected on the original mesh (before smoothing);
    /// this does not affect the vertices moved on other stages of the processing
    bool projectOnOriginalMesh;
    // TODO: onEdgeSplit
    // TODO: onEdgeDel
    // TODO: preCollapse
    /// callback to report algorithm progress and cancel it by user request
    MRProgressCallback progressCallback;
} MRRemeshSettings;

/// initializes a default instance
MRMESHC_API MRRemeshSettings mrRemeshSettingsNew( void );

/// Splits too long and eliminates too short edges from the mesh
MRMESHC_API bool mrRemesh( MRMesh* mesh, const MRRemeshSettings* settings );

/// creates a copy of a Mesh object
MRMESHC_API MRMesh* mrMeshCopy( const MRMesh* mesh );

/// constructs a mesh from vertex coordinates and a set of triangles with given ids
MRMESHC_API MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

/// constructs a mesh from vertex coordinates and a set of triangles with given ids;
/// unlike simple \ref mrMeshFromTriangles it tries to resolve non-manifold vertices by creating duplicate vertices
MRMESHC_API MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

/// constructs a mesh from point triples;
/// \param duplicateNonManifoldVertices = false, all coinciding points are given the same VertId in the result;
/// \param duplicateNonManifoldVertices = true, it tries to avoid non-manifold vertices by creating duplicate vertices with same coordinates
MRMESHC_API MRMesh* mrMeshNewFromPointTriples( const MRTriangle3f* posTriangles, size_t posTrianglesNum, bool duplicateNonManifoldVertices );

/// gets read-only access to the mesh vertex coordinates
MRMESHC_API const MRVector3f* mrMeshPoints( const MRMesh* mesh );

/// gets read-write access to the mesh vertex coordinates
MRMESHC_API MRVector3f* mrMeshPointsRef( MRMesh* mesh );

/// gets count of the mesh vertex coordinates
MRMESHC_API size_t mrMeshPointsNum( const MRMesh* mesh );

/// gets read-only access to the mesh topology object
MRMESHC_API const MRMeshTopology* mrMeshTopology( const MRMesh* mesh );

/// gets read-write access to the mesh topology object
MRMESHC_API MRMeshTopology* mrMeshTopologyRef( MRMesh* mesh );

/// applies given transformation to specified vertices
/// if region is NULL, all valid mesh vertices are used
MRMESHC_API void mrMeshTransform( MRMesh* mesh, const MRAffineXf3f* xf, const MRVertBitSet* region );

/// optional parameters for \ref mrMeshAddPartByMask
typedef struct MRMeshAddPartByMaskParameters
{
    /// if flipOrientation then every from triangle is inverted before adding
    bool flipOrientation;
    /// contours on this mesh that have to be stitched with
    const MREdgePath* thisContours;
    size_t thisContoursNum;
    /// contours on from mesh during addition
    const MREdgePath* fromContours;
    size_t fromContoursNum;
    // TODO: map
} MRMeshAddPartByMaskParameters;

/// appends mesh (from) in addition to this mesh: creates new edges, faces, verts and points
MRMESHC_API void mrMeshAddPartByMask( MRMesh* mesh, const MRMesh* from, const MRFaceBitSet* fromFaces, const MRMeshAddPartByMaskParameters* params );

/// deallocates a Mesh object
MRMESHC_API void mrMeshFree( MRMesh* mesh );

/// detects the format from file extension and loads mesh from it
/// if an error has occurred and errorStr is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API MRMesh* mrMeshLoadFromAnySupportedFormat( const char* file, MRString** errorStr );

/// stores reference on whole mesh (if region is NULL) or on its part (if region pointer is valid)
typedef struct MRMeshPart
{
    const MRMesh* mesh;
    const MRFaceBitSet* region;
} MRMeshPart;

/// detects the format from file extension and saves mesh to it
/// if an error has occurred and errorStr is not NULL, returns NULL and allocates an error message to errorStr
MRMESHC_API void mrMeshSaveToAnySupportedFormat( const MRMesh* mesh, const char* file, MRString** errorStr );

/// tightly packs all arrays eliminating lone edges and invalid faces and vertices
MRMESHC_API void mrMeshTopologyPack( MRMeshTopology* top );

/// returns cached set of all valid vertices
MRMESHC_API const MRVertBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top );

/// returns cached set of all valid faces
MRMESHC_API const MRFaceBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top );

/// returns three vertex ids for valid triangles (which can be accessed by FaceId),
/// vertex ids for invalid triangles are undefined, and shall not be read
MRMESHC_API MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top );

/// gets read-only access to the vertex triples of the triangulation
MRMESHC_API const MRThreeVertIds* mrTriangulationData( const MRTriangulation* tris );

/// gets total count of the vertex triples of the triangulation
MRMESHC_API size_t mrTriangulationSize( const MRTriangulation* tris );

/// deallocates the Triangulation object
MRMESHC_API void mrTriangulationFree( MRTriangulation* tris );

/// returns one edge with no valid left face for every boundary in the mesh
MRMESHC_API MREdgePath* mrMeshTopologyFindHoleRepresentiveEdges( const MRMeshTopology* top );

/// gets read-only access to the edges of the edge path
MRMESHC_API const MREdgeId* mrEdgePathData( const MREdgePath* ep );

/// gets total count of the edges of the edge path
MRMESHC_API size_t mrEdgePathSize( const MREdgePath* ep );

/// deallocates the EdgePath object
MRMESHC_API void mrEdgePathFree( MREdgePath* ep );

/// how to determine the sign of distances from a mesh
typedef enum MRSignDetectionMode
{
    /// unsigned distance, useful for bidirectional `Shell` offset
    MRSignDetectionModeUnsigned = 0,
    /// sign detection from OpenVDB library, which is good and fast if input geometry is closed
    MRSignDetectionModeOpenVDB,
    /// the sign is determined based on pseudonormal in closest mesh point (unsafe in case of self-intersections)
    MRSignDetectionModeProjectionNormal,
    /// ray intersection counter, significantly slower than ProjectionNormal and does not support holes in mesh
    MRSignDetectionModeWindingRule,
    /// computes winding number generalization with support of holes in mesh, slower than WindingRule
    MRSignDetectionModeHoleWindingRule
} MRSignDetectionMode;

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

/// returns closed loop of region boundary starting from given region boundary edge (region faces on the right, and not-region faces or holes on the left);
/// if more than two boundary edges connect in one vertex, then the function makes the most abrupt turn to left
MRMESHC_API MREdgeLoop* mrTrackRightBoundaryLoop( const MRMeshTopology* topology, MREdgeId e0, const MRFaceBitSet* region );

/// gets read-only access to the string data
MRMESHC_API const char* mrStringData( const MRString* str );

/// gets total length of the string
MRMESHC_API size_t mrStringSize( const MRString* str );

/// deallocates the string object
MRMESHC_API void mrStringFree( MRString* str );

/// parameters for \ref mrMakeTorus
typedef struct MRMakeTorusParameters
{
    float primaryRadius;
    float secondaryRadius;
    int primaryResolution;
    int secondaryResolution;
    // TODO: points
} MRMakeTorusParameters;

/// initializes a default instance
MRMESHC_API MRMakeTorusParameters mrMakeTorusParametersNew( void );

/// creates a mesh representing a torus
/// Z is symmetry axis of this torus
MRMESHC_API MRMesh* mrMakeTorus( const MRMakeTorusParameters* params );

/// (a, a, a)
MRMESHC_API MRVector3f mrVector3fDiagonal( float a );

/// (1, 0, 0)
MRMESHC_API MRVector3f mrVector3fPlusX( void );

/// (0, 1, 0)
MRMESHC_API MRVector3f mrVector3fPlusY( void );

/// (0, 0, 1)
MRMESHC_API MRVector3f mrVector3fPlusZ( void );

/// adds two vectors
MRMESHC_API MRVector3f mrVector3fAdd( const MRVector3f* a, const MRVector3f* b );

/// multiplies a vector by a scalar value
MRMESHC_API MRVector3f mrVector3fMulScalar( const MRVector3f* a, float b );

MR_EXTERN_C_END

