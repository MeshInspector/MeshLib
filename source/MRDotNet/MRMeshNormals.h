#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class MeshNormals
{
public:
    static VertNormals^ ComputePerVertNormals( Mesh^ mesh );
    static FaceNormals^ ComputePerFaceNormals( Mesh^ mesh );
};

MR_DOTNET_NAMESPACE_END
