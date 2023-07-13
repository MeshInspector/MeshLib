#pragma once
#ifdef _WIN32
#   ifdef MRCOMMONPLUGINS_EXPORT
#       define MRCOMMONPLUGINS_API __declspec(dllexport)
#   else
#       define MRCOMMONPLUGINS_API __declspec(dllimport)
#   endif
#   define MRCOMMONPLUGINS_CLASS
#else
#   define MRCOMMONPLUGINS_API __attribute__((visibility("default")))
// to fix undefined reference to `typeinfo/vtable
#   define MRCOMMONPLUGINS_CLASS __attribute__((visibility("default")))
#endif
