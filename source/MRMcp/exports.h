#pragma once

#ifdef _WIN32
#   ifdef MRMcp_EXPORTS
#       define MRMCP_API __declspec(dllexport)
#   else
#       define MRMCP_API __declspec(dllimport)
#   endif
#   define MRMCP_CLASS
#else
#   define MRMCP_API   __attribute__((visibility("default")))
#   ifdef __clang__
#       define MRMCP_CLASS __attribute__((type_visibility("default")))
#   else
#       define MRMCP_CLASS __attribute__((visibility("default")))
#   endif
#endif
