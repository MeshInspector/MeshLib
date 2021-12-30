#pragma once
#ifdef _WIN32
#   ifdef MREALGORITHMSPY_EXPORT
#       define MREALGORITHMSPY_API __declspec(dllexport)
#   else
#       define MREALGORITHMSPY_API __declspec(dllimport)
#   endif
#   define MREALGORITHMSPY_CLASS
#else
#   define MREALGORITHMSPY_API   __attribute__((visibility("default")))
// to fix undefined reference to `typeinfo/vtable
#   define MREALGORITHMSPY_CLASS __attribute__((visibility("default")))
#endif