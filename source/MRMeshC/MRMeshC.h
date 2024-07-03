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

typedef struct MRVector3f
{
    float x;
    float y;
    float z;
} MRVector3f;

typedef MRVector3f MRTriangle3f[3];

typedef struct MRMatrix3f
{
    MRVector3f x;
    MRVector3f y;
    MRVector3f z;
} MRMatrix3f;

typedef struct MRAffineXf3f
{
    MRMatrix3f A;
    MRVector3f b;
} MRAffineXf3f;

MRMESHC_API MRAffineXf3f mrAffineXf3fNew( void );

MRMESHC_API MRAffineXf3f mrAffineXf3fTranslation( const MRVector3f* b );

MRMESHC_API MRAffineXf3f mrAffineXf3fLinear( const MRMatrix3f* A );

MRMESHC_API MRAffineXf3f mrAffineXf3fMul( const MRAffineXf3f* a, const MRAffineXf3f* b );

MRMESHC_API const uint64_t* mrBitSetBlocks( const MRBitSet* bs );

MRMESHC_API size_t mrBitSetBlocksNum( const MRBitSet* bs );

MRMESHC_API size_t mrBitSetSize( const MRBitSet* bs );

MRMESHC_API bool mrBitSetEq( const MRBitSet* a, const MRBitSet* b );

MRMESHC_API void mrBitSetFree( MRBitSet* bs );

MRMESHC_API MRFaceBitSet* mrFaceBitSetCopy( const MRFaceBitSet* fbs );

MRMESHC_API void mrFaceBitSetFree( MRFaceBitSet* fbs );

typedef enum MRBooleanOperation
{
    MRBooleanOperationInsideA = 0,
    MRBooleanOperationInsideB,
    MRBooleanOperationOutsideA,
    MRBooleanOperationOutsideB,
    MRBooleanOperationUnion,
    MRBooleanOperationIntersection,
    MRBooleanOperationDifferenceBA,
    MRBooleanOperationDifferenceAB,
    MRBooleanOperationCount
} MRBooleanOperation;

MRMESHC_API MRMesh* mrMakeCube( const MRVector3f* size, const MRVector3f* base );

typedef struct MRMakeCylinderAdvancedParameters
{
    float radius0;
    float radius1;
    float startAngle;
    float arcSize;
    float length;
    int resolution;
} MRMakeCylinderAdvancedParameters;

MRMESHC_API MRMakeCylinderAdvancedParameters mrMakeCylinderAdvancedParametersNew( void );

MRMESHC_API MRMesh* mrMakeCylinderAdvanced( const MRMakeCylinderAdvancedParameters* params );

typedef struct MREdgeId { int id; } MREdgeId;
typedef struct MRFaceId { int id; } MRFaceId;
typedef struct MRVertId { int id; } MRVertId;

typedef MRVertId MRThreeVertIds[3];

MRMESHC_API MRMatrix3f mrMatrix3fIdentity( void );

MRMESHC_API MRMatrix3f mrMatrix3fRotationScalar( const MRVector3f* axis, float angle );

MRMESHC_API MRMatrix3f mrMatrix3fRotationVector( const MRVector3f* from, const MRVector3f* to );

MRMESHC_API MRMatrix3f mrMatrix3fMul( const MRMatrix3f* a, const MRMatrix3f* b );

typedef struct MRBooleanParameters
{
    const MRAffineXf3f* rigidB2A;
    bool mergeAllNonIntersectingComponents;
    MRProgressCallback cb;
} MRBooleanParameters;

typedef struct MRBooleanResult
{
    MRMesh* mesh;
    MRString* errorString;
} MRBooleanResult;

MRMESHC_API MRBooleanResult mrBoolean( const MRMesh* meshA, const MRMesh* meshB, MRBooleanOperation operation, const MRBooleanParameters* params );

typedef enum MRDecimateStrategy
{
    MRDecimateStrategyMinimizeError = 0,
    MRDecimateStrategyShortestEdgeFirst
} MRDecimateStrategy;

typedef struct MRDecimateSettings
{
    MRDecimateStrategy strategy;
    float maxError;
    float maxEdgeLen;
    float maxBdShift;
    float maxTriangleAspectRatio;
    float criticalTriAspectRatio;
    float tinyEdgeLength;
    float stabilizer;
    bool optimizeVertexPos;
    int maxDeletedVertices;
    int maxDeletedFaces;
    MRFaceBitSet* region;
    // TODO: notFlippable
    // TODO: edgesToCollapse
    // TODO: touchBdVertices
    // TODO: bdVerts
    float maxAngleChange;
    // TODO: preCollapse
    // TODO: adjustCollapse
    // TODO: onEdgeDel
    // TODO: vertForms
    bool packMesh;
    MRProgressCallback progressCallback;
    int subdivideParts;
    bool decimateBetweenParts;
    // TODO: partFaces
    int minFacesInPart;
} MRDecimateSettings;

MRMESHC_API MRDecimateSettings mrDecimateSettingsNew( void );

typedef struct MRDecimateResult
{
    int vertsDeleted;
    int facesDeleted;
    float errorIntroduced;
    bool cancelled;
} MRDecimateResult;

MRMESHC_API MRDecimateResult mrDecimateMesh( MRMesh* mesh, const MRDecimateSettings* settings );

typedef struct MRResolveMeshDegenSettings
{
    float maxDeviation;
    float tinyEdgeLength;
    float maxAngleChange;
    float criticalAspectRatio;
    float stabilizer;
    MRFaceBitSet* region;
} MRResolveMeshDegenSettings;

MRMESHC_API MRResolveMeshDegenSettings mrResolveMeshDegenSettingsNew( void );

MRMESHC_API bool mrResolveMeshDegenerations( MRMesh* mesh, const MRResolveMeshDegenSettings* settings );

typedef struct MRRemeshSettings
{
    float targetEdgeLen;
    int maxEdgeSplits;
    float maxAngleChangeAfterFlip;
    float maxBdShift;
    bool useCurvature;
    int finalRelaxIters;
    bool finalRelaxNoShrinkage;
    MRFaceBitSet* region;
    // TODO: notFlippable
    bool packMesh;
    bool projectOnOriginalMesh;
    // TODO: onEdgeSplit
    // TODO: onEdgeDel
    // TODO: preCollapse
    MRProgressCallback progressCallback;
} MRRemeshSettings;

MRMESHC_API MRRemeshSettings mrRemeshSettingsNew( void );

MRMESHC_API bool mrRemesh( MRMesh* mesh, const MRRemeshSettings* settings );

MRMESHC_API MRMesh* mrMeshCopy( const MRMesh* mesh );

MRMESHC_API MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

MRMESHC_API MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates, size_t vertexCoordinatesNum, const MRThreeVertIds* t, size_t tNum );

MRMESHC_API MRMesh* mrMeshNewFromPointTriples( const MRTriangle3f* posTriangles, size_t posTrianglesNum, bool duplicateNonManifoldVertices );

MRMESHC_API const MRVector3f* mrMeshPoints( const MRMesh* mesh );

MRMESHC_API MRVector3f* mrMeshPointsRef( MRMesh* mesh );

MRMESHC_API size_t mrMeshPointsNum( const MRMesh* mesh );

MRMESHC_API const MRMeshTopology* mrMeshTopology( const MRMesh* mesh );

MRMESHC_API MRMeshTopology* mrMeshTopologyRef( MRMesh* mesh );

MRMESHC_API void mrMeshTransform( MRMesh* mesh, const MRAffineXf3f* xf, const MRVertBitSet* region );

typedef struct MRMeshAddPartByMaskParameters
{
    bool flipOrientation;
    const MREdgePath* thisContours;
    size_t thisContoursNum;
    const MREdgePath* fromContours;
    size_t fromContoursNum;
    // TODO: map
} MRMeshAddPartByMaskParameters;

MRMESHC_API void mrMeshAddPartByMask( MRMesh* mesh, const MRMesh* from, const MRFaceBitSet* fromFaces, const MRMeshAddPartByMaskParameters* params );

MRMESHC_API void mrMeshFree( MRMesh* mesh );

MRMESHC_API MRMesh* mrMeshLoadFromAnySupportedFormat( const char* file, MRString** errorStr );

typedef struct MRMeshPart
{
    const MRMesh* mesh;
    const MRFaceBitSet* region;
} MRMeshPart;

MRMESHC_API void mrMeshSaveToAnySupportedFormat( const MRMesh* mesh, const char* file, MRString** errorStr );

MRMESHC_API void mrMeshTopologyPack( MRMeshTopology* top );

MRMESHC_API const MRVertBitSet* mrMeshTopologyGetValidVerts( const MRMeshTopology* top );

MRMESHC_API const MRFaceBitSet* mrMeshTopologyGetValidFaces( const MRMeshTopology* top );

MRMESHC_API MRTriangulation* mrMeshTopologyGetTriangulation( const MRMeshTopology* top );

MRMESHC_API const MRThreeVertIds* mrTriangulationData( const MRTriangulation* tris );

MRMESHC_API size_t mrTriangulationSize( const MRTriangulation* tris );

MRMESHC_API void mrTriangulationFree( MRTriangulation* tris );

MRMESHC_API MREdgePath* mrMeshTopologyFindHoleRepresentiveEdges( const MRMeshTopology* top );

MRMESHC_API const MREdgeId* mrEdgePathData( const MREdgePath* ep );

MRMESHC_API size_t mrEdgePathSize( const MREdgePath* ep );

MRMESHC_API void mrEdgePathFree( MREdgePath* ep );

typedef enum MRSignDetectionMode
{
    MRSignDetectionModeUnsigned = 0,
    MRSignDetectionModeOpenVDB,
    MRSignDetectionModeProjectionNormal,
    MRSignDetectionModeWindingRule,
    MRSignDetectionModeHoleWindingRule
} MRSignDetectionMode;

typedef struct MROffsetParameters
{
    float voxelSize;
    MRProgressCallback callBack;
    MRSignDetectionMode signDetectionMode;
    // TODO: fwn
    bool memoryEfficient;
} MROffsetParameters;

MRMESHC_API MROffsetParameters mrOffsetParametersNew( void );

MRMESHC_API MRMesh* mrOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, MRString** errorString );

MRMESHC_API MRMesh* mrDoubleOffsetMesh( MRMeshPart mp, float offsetA, float offsetB, const MROffsetParameters* params, MRString** errorString );

MRMESHC_API MRMesh* mrMcOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, MRString** errorString );

MRMESHC_API MRMesh* mrMcShellMeshRegion( const MRMesh* mesh, const MRFaceBitSet* region, float offset, const MROffsetParameters* params, MRString** errorString );

typedef enum MRGeneralOffsetParametersMode
{
    MRGeneralOffsetParametersModeSmooth = 0,
    MRGeneralOffsetParametersModeStandard,
    MRGeneralOffsetParametersModeSharpening
} MRGeneralOffsetParametersMode;

typedef struct MRGeneralOffsetParameters
{
    // TODO: outSharpEdges
    float minNewVertDev;
    float maxNewRank2VertDev;
    float maxNewRank3VertDev;
    float maxOldVertPosCorrection;
    MRGeneralOffsetParametersMode mode;
} MRGeneralOffsetParameters;

MRMESHC_API MRGeneralOffsetParameters mrGeneralOffsetParametersNew( void );

MRMESHC_API MRMesh* mrSharpOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

MRMESHC_API MRMesh* mrGeneralOffsetMesh( MRMeshPart mp, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

MRMESHC_API MRMesh* mrThickenMesh( const MRMesh* mesh, float offset, const MROffsetParameters* params, const MRGeneralOffsetParameters* generalParams, MRString** errorString );

MRMESHC_API MREdgeLoop* mrTrackRightBoundaryLoop( const MRMeshTopology* topology, MREdgeId e0, const MRFaceBitSet* region );

MRMESHC_API const char* mrStringData( const MRString* str );

MRMESHC_API size_t mrStringSize( const MRString* str );

MRMESHC_API void mrStringFree( MRString* str );

typedef struct MRMakeTorusParameters
{
    float primaryRadius;
    float secondaryRadius;
    int primaryResolution;
    int secondaryResolution;
    // TODO: points
} MRMakeTorusParameters;

MRMESHC_API MRMakeTorusParameters mrMakeTorusParametersNew( void );

MRMESHC_API MRMesh* mrMakeTorus( const MRMakeTorusParameters* params );

MRMESHC_API MRVector3f mrVector3fDiagonal( float a );

MRMESHC_API MRVector3f mrVector3fPlusX( void );

MRMESHC_API MRVector3f mrVector3fPlusY( void );

MRMESHC_API MRVector3f mrVector3fPlusZ( void );

MRMESHC_API MRVector3f mrVector3fAdd( const MRVector3f* a, const MRVector3f* b );

MRMESHC_API MRVector3f mrVector3fMulScalar( const MRVector3f* a, float b );

MR_EXTERN_C_END
