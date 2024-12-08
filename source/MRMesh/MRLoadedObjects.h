#pragma once

#include "MRMeshFwd.h"
#include <memory>

namespace MR
{

using ObjectPtr = std::shared_ptr<Object>;

/// result of loading (e.g. from a file) as one object (with possible subobjects)
struct LoadedObject
{
    ObjectPtr obj;
    std::string warnings; //either empty or ends with '\n'
};

/// result of mesh loading (e.g. from a file) as one object
struct LoadedObjectMesh
{
    std::shared_ptr<ObjectMesh> obj;
    std::string warnings; //either empty or ends with '\n'
};

/// result of loading (e.g. from a file) as a number of objects
struct LoadedObjects
{
    std::vector<ObjectPtr> objs;
    std::string warnings; //either empty or ends with '\n'
};

} //namespace MR
