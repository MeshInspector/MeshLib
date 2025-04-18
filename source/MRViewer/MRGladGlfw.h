#pragma once

#ifdef __EMSCRIPTEN__
#include <GLES3/gl3.h>
#else
#include <glad/glad.h>
#endif
#include <GLFW/glfw3.h>
#ifdef _WIN32
#undef APIENTRY
#endif

#ifndef __EMSCRIPTEN__
#define MR_GLSL_VERSION_LINE R"(#version 150)"
#else
#define MR_GLSL_VERSION_LINE R"(#version 300 es)"
#endif

namespace MR {

// Load OpenGL and its extensions
inline int loadGL()
{
#ifndef __EMSCRIPTEN__
#pragma warning(push)
#pragma warning(disable: 4191) //'type cast': unsafe conversion from 'GLFWglproc (__cdecl *)(const char *)' to 'GLADloadproc'
    // thread_local here - because we have two windows in two different threads (main window and splash)
    // so we need to be sure that each thread loaded gl (it can be called from GUI threads only)
    //
    // function is inline to be sure that each dll/so has its own instance of `loadRes`
    static thread_local auto loadRes = gladLoadGLLoader( ( GLADloadproc )glfwGetProcAddress );
    return loadRes;
#pragma warning(pop)
#else
return 1;
#endif
}

// finds power of 2 that represents given msaa number
// ==log2(msaa)
inline int getMSAAPow( int msaa )
{
    if ( msaa <= 1 )
        return 0;
    int i = 1;
    for ( ; i < 4; ++i )
    {
        if ( ( msaa & ( 1 << i ) ) != 0 )
            return i;
    }
    return i;
}

} //namespace MR
