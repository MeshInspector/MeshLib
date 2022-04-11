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
#ifndef MR_SCENEROOT_CONST
    MRMESH_API static Object& get();
    MRMESH_API static std::shared_ptr<Object>& getSharedPtr();

    MRMESH_API static void setScenePath( const std::filesystem::path& scenePath );
#endif
    MRMESH_API static const Object& constGet();
    MRMESH_API static std::shared_ptr<const Object> constGetSharedPtr();

    MRMESH_API static const std::filesystem::path& getScenePath();

private:
    static SceneRoot& instace_();
    SceneRoot();

    std::shared_ptr<Object> root_;

    // path to the recently opened scene
    std::filesystem::path scenePath_;
};
}
