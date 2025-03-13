#include "MRCudaMath.cuh"
#include "MRCudaMath.h"

#include "MRMesh/MRAffineXf3.h"

namespace MR::Cuda
{

float3 fromVec( const Vector3f& v )
{
    return { v.x, v.y, v.z };
}

int3 fromVec( const Vector3i& v )
{
    return { v.x, v.y, v.z };
}

Matrix4 fromXf( const MR::AffineXf3f& xf )
{
    if ( xf == AffineXf3f{} )
        return Matrix4 { .isIdentity = true };

    return {
        .x = fromVec( xf.A.x ),
        .y = fromVec( xf.A.y ),
        .z = fromVec( xf.A.z ),
        .b = fromVec( xf.b ),
        .isIdentity = false,
    };
}

} // namespace MR::Cuda
