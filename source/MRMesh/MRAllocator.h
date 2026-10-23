#pragma once

#include "MRMeshFwd.h"
#include <memory>

#if !defined( __EMSCRIPTEN__ ) && !defined( MR_MIMALLOC_OVERRIDE )
#include <mimalloc.h>
#if defined( _MSC_VER ) && !defined( MR_USE_CMAKE_CONFIGURE_FILE )
// the inline mi_stl_allocator calls mimalloc from every binary using MRMesh; MeshLib's CMake targets link it instead
#ifdef _DEBUG
#pragma comment( lib, "mimalloc-debug.dll.lib" )
#else
#pragma comment( lib, "mimalloc.dll.lib" )
#endif
#endif
#endif

namespace MR
{

/// allocator of MeshLib containers that benefit from mimalloc: mimalloc explicitly in MeshLib's own builds,
/// std::allocator where malloc is mimalloc already (Emscripten's -sMALLOC=mimalloc, MR_MIMALLOC_OVERRIDE executables)
#if !defined( __EMSCRIPTEN__ ) && !defined( MR_MIMALLOC_OVERRIDE )
template <typename T>
using Allocator = mi_stl_allocator<T>;
#else
template <typename T>
using Allocator = std::allocator<T>;
#endif

} // namespace MR
