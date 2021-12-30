#pragma once
#ifdef _WIN32
#   ifdef MREALGORITHMS_EXPORT
#       define MREALGORITHMS_API __declspec(dllexport)
#   else
#       define MREALGORITHMS_API __declspec(dllimport)
#   endif
#   define MREALGORITHMS_CLASS
#else
#   define MREALGORITHMS_API   __attribute__((visibility("default")))
// to fix undefined reference to `typeinfo/vtable
#   define MREALGORITHMS_CLASS __attribute__((visibility("default")))
#endif