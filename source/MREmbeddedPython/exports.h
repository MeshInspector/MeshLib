#pragma once

// see explanation in MRMesh/MRMeshFwd.h
#ifdef _WIN32
#   ifdef MREmbeddedPython_EXPORTS
#       define MREMBEDDEDPYTHON_API __declspec(dllexport)
#   else
#       define MREMBEDDEDPYTHON_API __declspec(dllimport)
#   endif
#   define MREMBEDDEDPYTHON_CLASS
#else
#   define MREMBEDDEDPYTHON_API   __attribute__((visibility("default")))
#   ifdef __clang__
#       define MREMBEDDEDPYTHON_CLASS __attribute__((type_visibility("default")))
#   else
#       define MREMBEDDEDPYTHON_CLASS __attribute__((visibility("default")))
#   endif
#endif
