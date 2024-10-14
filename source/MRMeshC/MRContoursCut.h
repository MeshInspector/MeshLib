#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRIntersectionContour.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

typedef enum MROneMeshIntersectionVariantIndex
{
    MROneMeshIntersectionVariantIndexFace,
    MROneMeshIntersectionVariantIndexEdge,
    MROneMeshIntersectionVariantIndexVertex
}
MROneMeshIntersectionVariantIndex;

// Simple point on mesh, represented by primitive id and coordinate in mesh space
typedef struct MROneMeshIntersection
{
    union
    {
        MRFaceId face;
        MREdgeId edge;
        MRVertId vertex;
    } primitiveId;
    unsigned char primitiveIdIndex;
    MRVector3f coordinate;
}
MROneMeshIntersection;

MR_VECTOR_DECL( OneMeshIntersection )

// One contour on mesh
typedef struct MROneMeshContour
{
    MRVectorOneMeshIntersection intersections;
    bool closed;
}
MROneMeshContour;

typedef struct MROneMeshContours MROneMeshContours;

/// gets the contours' value at index
MRMESHC_API const MROneMeshContour mrOneMeshContoursGet( const MROneMeshContours* contours, size_t index );

/// gets the contours' size
MRMESHC_API size_t mrOneMeshContoursSize( const MROneMeshContours* contours );

/// deallocates the OneMeshContours object
MRMESHC_API void mrOneMeshContoursFree( MROneMeshContours* contours );

// Converts ordered continuous contours of two meshes to OneMeshContours
// converters is required for better precision in case of degenerations
// note that contours should not have intersections
MRMESHC_API MROneMeshContours* mrGetOneMeshIntersectionContours( const MRMesh* meshA, const MRMesh* meshB,
                                                                 const MRContinuousContours* contours,
                                                                 bool getMeshAIntersections,
                                                                 const MRCoordinateConverters* converters,
                                                                 const MRAffineXf3f* rigidB2A );

MR_EXTERN_C_END
