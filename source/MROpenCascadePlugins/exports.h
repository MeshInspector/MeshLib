#pragma once
#ifdef _WIN32
#   ifdef MROPENCASCADEPLUGINS_EXPORT
#       define MROPENCASCADEPLUGINS_API __declspec(dllexport)
#   else
#       define MROPENCASCADEPLUGINS_API __declspec(dllimport)
#   endif
#   define MROPENCASCADEPLUGINS_CLASS
#else
#   define MROPENCASCADEPLUGINS_API   __attribute__((visibility("default")))
// to fix undefined reference to `typeinfo/vtable
#   define MROPENCASCADEPLUGINS_CLASS __attribute__((visibility("default")))
#endif

namespace MR
{

MROPENCASCADEPLUGINS_API void loadMROpenCascadePluginsDll();

} // namespace MR