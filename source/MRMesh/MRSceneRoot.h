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

private:
    static SceneRoot& instace_();
    SceneRoot();

    std::shared_ptr<Object> root_;
};
}
