#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace MR::McpGateway
{

/// Spawns @p exe with @p args as a detached child process and returns immediately.
/// The child survives this process exiting (Windows: DETACHED_PROCESS; POSIX: setsid + double-fork).
/// Returns true if the child was launched successfully (does not verify it then ran to completion).
bool spawnDetached( const std::filesystem::path& exe, const std::vector<std::string>& args );

/// Spawns @p exe with @p args as an attached child process and blocks until it exits
/// or @p timeout elapses. On timeout the child is force-killed. Returns true iff the
/// child ran to completion within the timeout (regardless of its exit code — caller
/// decides what counts as success based on side effects, e.g. a written cache file).
bool spawnAndWait( const std::filesystem::path& exe, const std::vector<std::string>& args,
                   std::chrono::seconds timeout );

} // namespace MR::McpGateway
