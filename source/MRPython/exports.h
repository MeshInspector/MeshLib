#pragma once

#ifdef _WIN32
#   ifdef MRPYTHON_EXPORT
#       define MRPYTHON_API __declspec(dllexport)
#   else
#       define MRPYTHON_API __declspec(dllimport)
#   endif
#   define MRPYTHON_CLASS
#else
#   define MRPYTHON_API   __attribute__((visibility("default")))
#   define MRPYTHON_CLASS __attribute__((visibility("default")))
#endif
