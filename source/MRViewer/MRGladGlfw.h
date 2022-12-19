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

namespace MR {

// Load OpenGL and its extensions
inline int loadGL()
{
#ifndef __EMSCRIPTEN__
#pragma warning(push)
#pragma warning(disable: 4191) //'type cast': unsafe conversion from 'GLFWglproc (__cdecl *)(const char *)' to 'GLADloadproc'
    static auto loadRes = gladLoadGLLoader( ( GLADloadproc )glfwGetProcAddress );
    return loadRes;
#pragma warning(pop)
#else
return 1;
#endif
}

} //namespace MR
