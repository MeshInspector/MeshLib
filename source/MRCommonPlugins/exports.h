#pragma once

// see explanation in MRMesh/MRMeshFwd.h
#ifdef _WIN32
#   ifdef MRCommonPlugins_EXPORTS
#       define MRCOMMONPLUGINS_API __declspec(dllexport)
#   else
#       define MRCOMMONPLUGINS_API __declspec(dllimport)
#   endif
#   define MRCOMMONPLUGINS_CLASS
#else
#   define MRCOMMONPLUGINS_API __attribute__((visibility("default")))
#   ifdef __clang__
#       define MRCOMMONPLUGINS_CLASS __attribute__((type_visibility("default")))
#   else
#       define MRCOMMONPLUGINS_CLASS __attribute__((visibility("default")))
#   endif
#endif
