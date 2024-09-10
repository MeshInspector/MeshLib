#pragma once

#ifdef _WIN32
#   ifdef MRIOEXTRAS_EXPORT
#       define MRIOEXTRAS_API __declspec(dllexport)
#   else
#       define MRIOEXTRAS_API __declspec(dllimport)
#   endif
#   define MRIOEXTRAS_CLASS
#else
#   define MRIOEXTRAS_API   __attribute__((visibility("default")))
#   define MRIOEXTRAS_CLASS __attribute__((visibility("default")))
#endif
