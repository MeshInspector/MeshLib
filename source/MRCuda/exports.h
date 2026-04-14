#pragma once

// see explanation in MRMesh/MRMeshFwd.h
#ifdef _WIN32
#   ifdef MRCuda_EXPORTS
#       define MRCUDA_API __declspec(dllexport)
#   else
#       define MRCUDA_API __declspec(dllimport)
#   endif
#   define MRCUDA_CLASS
#else
#   define MRCUDA_API   __attribute__((visibility("default")))
#   ifdef __clang__
#       define MRCUDA_CLASS __attribute__((type_visibility("default")))
#   else
#       define MRCUDA_CLASS __attribute__((visibility("default")))
#   endif
#endif