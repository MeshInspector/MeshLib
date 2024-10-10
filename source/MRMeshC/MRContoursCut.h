#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRIntersectionContour.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

/// ...
typedef enum MROneMeshIntersectionVariantIndex
{
    MROneMeshIntersectionVariantIndexFace,
    MROneMeshIntersectionVariantIndexEdge,
    MROneMeshIntersectionVariantIndexVertex
}
MROneMeshIntersectionVariantIndex;

/// ...
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

/// ...
typedef struct MROneMeshContour
{
    MRVectorOneMeshIntersection intersections;
    bool closed;
}
MROneMeshContour;

/// ...
typedef struct MROneMeshContours MROneMeshContours;

/// ...
MRMESHC_API const MROneMeshContour mrOneMeshContoursGet( const MROneMeshContours* contours, size_t index );

/// ...
MRMESHC_API size_t mrOneMeshContoursSize( const MROneMeshContours* contours );

/// ...
MRMESHC_API void mrOneMeshContoursFree( MROneMeshContours* contours );

/// ...
MRMESHC_API MROneMeshContours* mrGetOneMeshIntersectionContours( const MRMesh* meshA, const MRMesh* meshB,
                                                                 const MRContinuousContours* contours,
                                                                 bool getMeshAIntersections,
                                                                 const MRCoordinateConverters* converters,
                                                                 const MRAffineXf3f* rigidB2A );

MR_EXTERN_C_END
