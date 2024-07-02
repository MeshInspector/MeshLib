#include "MRCube.h"

#include "MRMesh/MRCube.h"
#include "MRMesh/MRMesh.h"

using namespace MR;

MRMesh* mrMakeCube( const MRVector3f* size_, const MRVector3f* base_ )
{
    const auto& size = *reinterpret_cast<const Vector3f*>( size_ );
    const auto& base = *reinterpret_cast<const Vector3f*>( base_ );

    auto res = makeCube( size, base );
    return reinterpret_cast<MRMesh*>( new Mesh( std::move( res ) ) );
}
