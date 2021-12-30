#pragma once
#ifdef _WIN32
#   ifdef MRMESHPY_EXPORT
#       define MRMESHPY_API __declspec(dllexport)
#   else
#       define MRMESHPY_API __declspec(dllimport)
#   endif
#   define MRMESHPY_CLASS
#else
#   define MRMESHPY_API   __attribute__((visibility("default")))
// to fix undefined reference to `typeinfo/vtable
#   define MRMESHPY_CLASS __attribute__((visibility("default")))
#endif