#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

#include <iostream>

int main()
{
    std::filesystem::path inFilePath = "mesh.stl";
    auto loadRes = MR::MeshLoad::fromAnySupportedFormat( inFilePath );
    if ( loadRes.has_value() )
    {
        std::filesystem::path outFilePath = "mesh.ply";
        auto saveRes = MR::MeshSave::toAnySupportedFormat( loadRes.value(), outFilePath );
        if ( !saveRes.has_value() )
            std::cerr << saveRes.error() << std::endl;
    }
    else
        std::cerr << loadRes.error() << std::endl;
    return 0;
}
