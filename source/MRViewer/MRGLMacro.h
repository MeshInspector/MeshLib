#pragma once

#if !defined( NDEBUG ) && !defined( MR_GL_NO_LOGGING )
#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRFinally.h"

// Not using `__VA_OPT__(,)` here to support legacy MSVC preprocessor.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

#define GL_EXEC( ... ) \
    ( \
        void(MR::detail::ScopeGuard([] \
        { \
            auto error = glGetError();\
            if ( error != 0 )\
                spdlog::warn("GL error: {} In file: {} Line: {}", error , __FILE__ , __LINE__ );\
        })) \
        ,##__VA_ARGS__\
    )

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#else
#define GL_EXEC( ... ) __VA_ARGS__
#endif
