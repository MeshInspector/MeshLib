#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace MR::McpGateway
{

// UTF-8 std::string -> std::filesystem::path. On Windows this is non-trivial:
// `path( std::string )` interprets the bytes as the system code page, which silently
// loses any character not representable in the current locale (e.g. CJK on US-Windows).
// On POSIX, paths are UTF-8 by convention so this is essentially a passthrough.
std::filesystem::path pathFromUtf8( const std::string& s );

// std::filesystem::path -> UTF-8 std::string. Mirror of pathFromUtf8 — on Windows,
// `path::string()` narrows via the system code page, which corrupts non-ASCII chars
// before we hand the string off to spawn args / log output / cerr.
std::string pathToUtf8( const std::filesystem::path& p );

#ifdef _WIN32
// Re-decodes argv from `GetCommandLineW` so that non-ASCII paths and arguments
// survive a launch from the command line. The CRT-supplied `argv` in `main(int,char**)`
// is system-codepage on Windows, so any character outside the locale (e.g. Cyrillic
// on a US-Windows install) becomes `?` before we even parse the flags.
std::vector<std::string> getUtf8Argv();
#endif

} // namespace MR::McpGateway
