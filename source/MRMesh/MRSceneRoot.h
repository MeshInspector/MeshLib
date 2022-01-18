#pragma once
#include "MRMeshFwd.h"
#include "MRObject.h"
#include <memory>

namespace MR
{

// Singleton to store scene root object
class SceneRoot
{
public:
    MRMESH_API static Object& get();
    MRMESH_API static std::shared_ptr<Object>& getSharedPtr();

    MRMESH_API static bool isScenePathEmpty();
    MRMESH_API static void clearScenePath();

    MRMESH_API static std::filesystem::path getScenePath();
    MRMESH_API static std::shared_ptr<std::filesystem::path>& getScenePathSharedPtr();

    MRMESH_API static std::string getSceneFileName();

    MRMESH_API static void setScenePath( const std::filesystem::path& scenePath );

private:
    static SceneRoot& instace_();
    SceneRoot();

    std::shared_ptr<Object> root_;
    std::shared_ptr<std::filesystem::path> scenePath_;
};
}
