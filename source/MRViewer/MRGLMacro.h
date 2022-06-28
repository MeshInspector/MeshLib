#pragma once
#ifndef NDEBUG
#include "MRPch/MRSpdlog.h"
#define GL_EXEC( func )\
func; \
{\
    auto error = glGetError();\
    if ( error != 0 )\
        spdlog::warn("GL error: {} In file: {} Line: {}", error , __FILE__ , __LINE__ );\
}
#else
#define GL_EXEC( func ) func;
#endif