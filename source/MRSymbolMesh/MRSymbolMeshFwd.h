#pragma once

#ifdef _WIN32
#   ifdef MRSymbolMesh_EXPORTS
#       define MRSYMBOLMESH_API __declspec(dllexport)
#   else
#       define MRSYMBOLMESH_API __declspec(dllimport)
#   endif
#   define MRSYMBOLMESH_CLASS
#else
#   define MRSYMBOLMESH_API   __attribute__((visibility("default")))
#   define MRSYMBOLMESH_CLASS __attribute__((visibility("default")))
#endif

#include <MRMesh/MRMeshFwd.h>
