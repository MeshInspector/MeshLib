#pragma once
#include "MRViewerFwd.h"
#include <optional>
#include <string>

namespace MR
{

/// a known issue of an OpenGL driver
struct GLDriverIssue
{
    /// stable identifier of the issue, the same for every driver version affected by it,
    /// usually the upstream bug tracker reference
    std::string id;
    /// human-readable description of the issue together with its remedy
    std::string description;
};

/// returns the known issue of the current OpenGL driver, or nullopt when no issue is known for it
MRVIEWER_API std::optional<GLDriverIssue> glDriverKnownIssues();

}
