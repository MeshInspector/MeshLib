#pragma once

#include "MRMeshFwd.h"
#include <memory>

// mr_enable_mimalloc executables forward malloc to the same mimalloc; MRMESH_NO_MIMALLOC, see cmake/Modules/Mimalloc.cmake
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_MIMALLOC )
#define MR_ALLOCATOR_IS_MIMALLOC
#endif

#ifdef MR_ALLOCATOR_IS_MIMALLOC
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

/// allocator of MeshLib containers that benefit from mimalloc: mimalloc explicitly,
/// std::allocator in Emscripten builds, where malloc is mimalloc already (-sMALLOC=mimalloc), and with MRMESH_NO_MIMALLOC
#ifdef MR_ALLOCATOR_IS_MIMALLOC
template <typename T>
using Allocator = mi_stl_allocator<T>;
#else
template <typename T>
using Allocator = std::allocator<T>;
#endif

} // namespace MR
