#pragma once
#ifndef MESHLIB_NO_MCP

#include "MRMesh/MRExpected.h"

#include <cstdint>
#include <memory>

namespace MR
{

class Object;

namespace Mcp
{

// Pointer -> id (as a JSON number). Matches the `##<id>` suffix in ImGui labels for the same object.
uint64_t idOf( const Object* obj );

// Resolve a raw-pointer id (as returned by scene.listObjectTree) to a shared_ptr<Object>, verifying
// the object is still reachable from SceneRoot. Defends against stale ids into freed memory.
Expected<std::shared_ptr<Object>> resolveId( uint64_t id );

} // namespace Mcp

} // namespace MR

#endif
