#include "MRGladGlfw.h"
#include "MRPch/MRWasm.h"

// Modifier for shortcuts
// Some shortcuts still use GLFW_MOD_CONTROL on Mac to avoid conflict with system shortcuts
#if !defined( __APPLE__ )
#define MR_CONTROL_OR_SUPER GLFW_MOD_CONTROL
#else
#define MR_CONTROL_OR_SUPER GLFW_MOD_SUPER
#endif

namespace MR
{

int getGlfwModPrimaryCtrl()
{
#ifndef __EMSCRIPTEN__
    return MR_CONTROL_OR_SUPER;
#else
    static const auto isMac = bool( EM_ASM_INT( return is_mac() ) );
    if ( isMac )
        return GLFW_MOD_SUPER;
    else
        return GLFW_MOD_CONTROL;
#endif
}

const char* getSuperModName()
{
    static const char* superName = [] ()->const char*
    {
#ifdef _WIN32
        return "Win";
#elif defined(__APPLE__)
        return "Command";
#elif defined(__EMSCRIPTEN__)
        if ( bool( EM_ASM_INT( return is_mac() ) ) )
            return "Command";
        else
            return "Super"; // we cannot distinguish windows and linux now
#else
        return "Super";
#endif
    }( );
    return superName;
}

const char* getAltModName()
{
    static const char* altName = [] ()->const char*
    {
#ifdef __EMSCRIPTEN__
        if ( bool( EM_ASM_INT( return is_mac() ) ) )
            return "Option";
        else
            return "Alt";
#elif defined(__APPLE__)
        return "Option";
#else
        return "Alt";
#endif
    }( );
    return altName;
}

}