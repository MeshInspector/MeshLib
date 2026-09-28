#pragma once

#include "MRViewerFwd.h"

#include <string>
#include <vector>

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

/// returns the known issues of the current OpenGL driver
MRVIEWER_API std::vector<GLDriverIssue> glDriverKnownIssues();

} // namespace MR
