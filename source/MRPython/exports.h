#pragma once

// see explanation in MRMesh/MRMeshFwd.h
#ifdef _WIN32
#   ifdef MRPython_EXPORTS
#       define MRPYTHON_API __declspec(dllexport)
#   else
#       define MRPYTHON_API __declspec(dllimport)
#   endif
#   define MRPYTHON_CLASS
#else
#   define MRPYTHON_API   __attribute__((visibility("default")))
#   ifdef __clang__
#       define MRPYTHON_CLASS __attribute__((type_visibility("default")))
#   else
#       define MRPYTHON_CLASS __attribute__((visibility("default")))
#   endif
#endif
