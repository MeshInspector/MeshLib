#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class MeshNormals
{
public:
    /// returns a list with vertex normals in every element for valid mesh vertices
    static VertNormals^ ComputePerVertNormals( Mesh^ mesh );
    /// returns a list with face normals in every element for valid mesh faces
    static FaceNormals^ ComputePerFaceNormals( Mesh^ mesh );
};

MR_DOTNET_NAMESPACE_END
