#pragma once

// see explanation in MRMesh/MRMeshFwd.h
#ifdef _WIN32
#   ifdef MRSymbolMesh_EXPORTS
#       define MRSYMBOLMESH_API __declspec(dllexport)
#   else
#       define MRSYMBOLMESH_API __declspec(dllimport)
#   endif
#   define MRSYMBOLMESH_CLASS
#else
#   define MRSYMBOLMESH_API   __attribute__((visibility("default")))
#   ifdef __clang__
#       define MRSYMBOLMESH_CLASS __attribute__((type_visibility("default")))
#   else
#       define MRSYMBOLMESH_CLASS __attribute__((visibility("default")))
#   endif
#endif

#include <MRMesh/MRMeshFwd.h>
