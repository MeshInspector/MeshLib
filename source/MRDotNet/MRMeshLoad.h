#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public value struct ObjLoadSettings
{
    /// if true then vertices will be returned relative to some transformation to avoid precision loss
    bool customXf;

    /// if true, the number of skipped faces (faces than can't be created) will be counted
    bool countSkippedFaces;
};

public value struct NamedMesh
{
    System::String^ name;
    Mesh^ mesh;
    AffineXf3f^ xf;
    int skippedFaceCount;
    int duplicatedVertexCount;
};

public ref class MeshLoad
{
public:
    static List<NamedMesh>^ FromSceneObjFile( System::String^ path, bool combineAllObjects, ObjLoadSettings settings );
};

MR_DOTNET_NAMESPACE_END