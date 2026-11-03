#pragma once
#ifndef MESHLIB_NO_MCP

#include "MRMesh/MRExpected.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace MR
{

class Object;

namespace Mcp
{

// Pointer -> id (as a JSON number). Matches the `##<id>` suffix in ImGui labels for the same object.
uint64_t idOf( const Object* obj );

// Resolve a raw-pointer id (as returned by scene_listObjectTree) to a shared_ptr<Object>, verifying
// the object is still reachable from SceneRoot. Defends against stale ids into freed memory.
Expected<std::shared_ptr<Object>> resolveId( uint64_t id );

// How long a tool waits for the main thread, e.g. while it runs a long action,
// so that the agent gets an explanation instead of a timeout in its MCP client
inline constexpr std::chrono::seconds cMainThreadTimeout{ 20 };

// The error for a main thread that did not respond within `cMainThreadTimeout`
std::runtime_error busyError( std::string_view what );

// `CommandLoop::runCommandFromGUIThread`, but throws `busyError` if the main thread does not start `func` in time
void runOnMainThread( const std::function<void()>& func );

// `skipFramesAfterInput`, then `after` on the main thread, limited to `cMainThreadTimeout` in total;
// returns a message for the agent if the main thread is still busy, likely with the action the input started
// (then `after` did not run); a tool whose input cannot start a long action may ignore it, the next call reports it
[[nodiscard]] std::string waitFramesAfterInput( const std::function<void()>& after = {} );

// The result of a tool that sends an input: `{}`, or `{busy: <message>}` from `waitFramesAfterInput`
nlohmann::json inputResult( std::string busy );

} // namespace Mcp

} // namespace MR

#endif
