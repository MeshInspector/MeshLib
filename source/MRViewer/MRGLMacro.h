#pragma once

#ifndef NDEBUG
#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRFinally.h"
#define GL_EXEC( ... ) \
    [&]() -> decltype(auto) \
    { \
        MR_FINALLY{ \
            auto error = glGetError();\
            if ( error != 0 )\
                spdlog::warn("GL error: {} In file: {} Line: {}", error , __FILE__ , __LINE__ );\
        }; \
        return __VA_ARGS__;\
    }()
#else
#define GL_EXEC( ... ) __VA_ARGS__
#endif