#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public value struct NamedMeshXf
{
    System::String^ name;
    Mesh^ mesh;
    AffineXf3f^ toWorld;
};

public ref class MeshSave
{
public:
    static void SceneToObj( List<NamedMeshXf>^ meshes, System::String^ path );
};

MR_DOTNET_NAMESPACE_END