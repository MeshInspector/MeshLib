#pragma once

#ifdef __EMSCRIPTEN__
#include "MRPch/MRWasm.h"

namespace MR
{

/// Return the pointer size in bytes
EMSCRIPTEN_KEEPALIVE int emsGetPointerSize()
{
    return sizeof (void*);
}

} // namespace MR
#endif
