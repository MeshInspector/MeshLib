#include "MRMeshNormals.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRMeshNormals.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_VECTOR( FaceNormals )

MRFaceNormals* mrComputePerFaceNormals( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW_VECTOR( computePerFaceNormals( mesh ).vec_ );
}

MRVertNormals* mrComputePerVertNormals( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW_VECTOR( computePerVertNormals( mesh ) );
}

MRVertNormals* mrComputePerVertPseudoNormals( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW_VECTOR( computePerVertPseudoNormals( mesh ) );
}

MRMeshNormals mrComputeMeshNormals( const MRMesh* mesh_ )
{
    ARG( mesh );
    auto result = computeMeshNormals( mesh );
    return {
        .faceNormals = auto_cast( NEW_VECTOR( std::move( result.faceNormals ) ) ),
        .vertNormals = auto_cast( NEW_VECTOR( std::move( result.vertNormals ) ) ),
    };
}
