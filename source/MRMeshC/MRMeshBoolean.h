#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRBooleanOperation.h"

MR_EXTERN_C_BEGIN

/// optional parameters for \ref mrBoolean
typedef struct MRBooleanParameters
{
    /// Transform from mesh `B` space to mesh `A` space
    const MRAffineXf3f* rigidB2A;
    /// Optional output structure to map mesh `A` and mesh `B` topology to result mesh topology
    MRBooleanResultMapper* mapper;
    // TODO: outPreCutA
    // TODO: outPreCutB
    // TODO: outCutEdges
    /// By default produce valid operation on disconnected components
    /// if set merge all non-intersecting components
    bool mergeAllNonIntersectingComponents;
    /// Progress callback
    MRProgressCallback cb;
} MRBooleanParameters;

/// initializes a default instance
MRMESHC_API MRBooleanParameters mrBooleanParametersNew( void );

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

MR_EXTERN_C_END
