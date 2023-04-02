#pragma once
#ifdef _WIN32
#   ifdef MRCUDA_EXPORT
#       define MRCUDA_API __declspec(dllexport)
#   else
#       define MRCUDA_API __declspec(dllimport)
#   endif
#   define MRCUDA_CLASS
#else
#   define MRCUDA_CLASS   __attribute__((visibility("default")))
// to fix undefined reference to `typeinfo/vtable
#   define MRCUDA_API __attribute__((visibility("default")))
#endif