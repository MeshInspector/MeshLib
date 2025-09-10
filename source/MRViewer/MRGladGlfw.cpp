#include "MRGladGlfw.h"
#include "MRPch/MRWasm.h"

namespace MR
{

int glfwModCtrlOrSupper()
{
#ifndef __EMSCRIPTEN__
    return MR_CONTROL_OR_SUPER;
#else
    const auto isMac = bool( EM_ASM_INT( return is_mac() ) );
    if ( isMac )
        return GLFW_MOD_SUPER;
    else
        return GLFW_MOD_CONTROL;
#endif
}

}