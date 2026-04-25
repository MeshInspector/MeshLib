#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace MR::McpGateway
{

/// Spawns @p exe with @p args as a detached child process and returns immediately.
/// The child survives this process exiting (Windows: DETACHED_PROCESS; POSIX: setsid + double-fork).
/// Returns true if the child was launched successfully (does not verify it then ran to completion).
bool spawnDetached( const std::filesystem::path& exe, const std::vector<std::string>& args );

} // namespace MR::McpGateway
