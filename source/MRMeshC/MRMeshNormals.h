#pragma once

#include "MRMeshFwd.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

typedef MRVectorVector3f MRFaceNormals;
typedef MRVectorVector3f MRVertNormals;

typedef struct MRMeshNormals
{
    MRFaceNormals* faceNormals;
    MRVertNormals* vertNormals;
} MRMeshNormals;

/// returns a vector with face-normal in every element for valid mesh faces
MRMESHC_API MRFaceNormals* mrComputePerFaceNormals( const MRMesh* mesh );

/// returns a vector with vertex normals in every element for valid mesh vertices
MRMESHC_API MRVertNormals* mrComputePerVertNormals( const MRMesh* mesh );

/// returns a vector with vertex pseudonormals in every element for valid mesh vertices
/// see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.9173&rep=rep1&type=pdf
MRMESHC_API MRVertNormals* mrComputePerVertPseudoNormals( const MRMesh* mesh );

/// computes both per-face and per-vertex normals more efficiently then just calling both previous functions
MRMESHC_API MRMeshNormals mrComputeMeshNormals( const MRMesh* mesh );

MR_EXTERN_C_END
