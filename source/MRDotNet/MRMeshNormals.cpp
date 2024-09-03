#include "MRMeshNormals.h"
#include "MRMesh.h"
#include "MRVector3.h"

#pragma managed( push, off )
#include <MRMesh/MRMeshNormals.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

VertNormals^ ComputePerVertNormals( Mesh^ mesh )
{
    auto nativeRes = MR::computePerFaceNormals( *mesh->getMesh() );
    VertNormals^ res = gcnew VertNormals( int( nativeRes.size() ) );
    for ( size_t i = 0; i < nativeRes.size(); i++ )
        res->Add( gcnew Vector3f( new MR::Vector3f( std::move( nativeRes.vec_[i] ) ) ) );
    
    return res;
}

FaceNormals^ ComputePerFaceNormals( Mesh^ mesh )
{
    auto nativeRes = MR::computePerFaceNormals( *mesh->getMesh() );
    FaceNormals^ res = gcnew FaceNormals( int( nativeRes.size() ) );
    for ( size_t i = 0; i < nativeRes.size(); i++ )
        res->Add( gcnew Vector3f( new MR::Vector3f( std::move( nativeRes.vec_[i] ) ) ) );

    return res;
}

MR_DOTNET_NAMESPACE_END
