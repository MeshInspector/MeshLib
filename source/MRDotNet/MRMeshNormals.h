#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

ref class MeshNormals
{
    static VertNormals^ ComputePerVertNormals( Mesh^ mesh );
    static FaceNormals^ ComputePerFaceNormals( Mesh^ mesh );
};

MR_DOTNET_NAMESPACE_END
