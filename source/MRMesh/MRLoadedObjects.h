#pragma once

#include "MRMeshFwd.h"
#include <memory>

namespace MR
{

class ObjectVoxels;
using ObjectPtr = std::shared_ptr<Object>;

/// result of loading (e.g. from a file) as one object (with possible subobjects)
template<typename ObjectT = Object>
struct LoadedObjectT
{
    std::shared_ptr<ObjectT> obj;
    std::string warnings; //either empty or ends with '\n'
};

using LoadedObject =       LoadedObjectT<Object>;
using LoadedObjectMesh =   LoadedObjectT<ObjectMesh>;
using LoadedObjectPoints = LoadedObjectT<ObjectPoints>;
using LoadedObjectLines  = LoadedObjectT<ObjectLines>;
using LoadedObjectVoxels = LoadedObjectT<ObjectVoxels>;

/// result of loading (e.g. from a file) as a number of objects
struct LoadedObjects
{
    std::vector<ObjectPtr> objs;
    std::string warnings; //either empty or ends with '\n'
};

enum class LoadedObjectsErrorKind
{
    generic,
    tryAnotherFormat, // The file clearly uses some other format.
};

struct LoadedObjectsError
{
    std::string message;
    LoadedObjectsErrorKind kind = LoadedObjectsErrorKind::generic;
};

} //namespace MR
