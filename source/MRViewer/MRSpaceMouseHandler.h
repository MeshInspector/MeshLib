#pragma once

#include "MRMesh/MRMeshFwd.h"
#include <functional>
#include <string>

namespace MR::SpaceMouse
{

/// base class for handler of spacemouse devices
class Handler
{
public:
    virtual ~Handler() = default;

    /// initialize device
    [[nodiscard]] virtual bool initialize() = 0;

    /// handle device state and call Viewer signals
    virtual void handle() = 0;
};

} //namespace MR
