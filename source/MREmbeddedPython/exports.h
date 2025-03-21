#pragma once

#ifdef _WIN32
#   ifdef MREmbeddedPython_EXPORTS
#       define MREMBEDDEDPYTHON_API __declspec(dllexport)
#   else
#       define MREMBEDDEDPYTHON_API __declspec(dllimport)
#   endif
#   define MREMBEDDEDPYTHON_CLASS
#else
#   define MREMBEDDEDPYTHON_API   __attribute__((visibility("default")))
#   define MREMBEDDEDPYTHON_CLASS __attribute__((visibility("default")))
#endif
