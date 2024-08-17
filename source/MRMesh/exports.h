#pragma once

#ifdef _WIN32
#   ifdef MRMESH_EXPORT
#       define MRMESH_API __declspec(dllexport)
#   else
#       define MRMESH_API __declspec(dllimport)
#   endif
#   define MRMESH_CLASS
#else
#   define MRMESH_API   __attribute__((visibility("default")))
// to fix undefined reference to `typeinfo/vtable`
// Also it's important to use this on any type for which `typeid` is used in multiple shared libraries, and then passed across library boundaries.
//   Otherwise on Mac the resulting typeids will incorrectly compare not equal.
#   define MRMESH_CLASS __attribute__((visibility("default")))
#endif
