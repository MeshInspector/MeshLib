#pragma once

#include "MRMesh/MRMeshFwd.h"
#include <functional>
#include <string>

namespace MR
{

/// base class for handler of spacemouse devices
class SpaceMouseHandler
{
public:
    virtual ~SpaceMouseHandler() = default;

    /// initialize device
    /// \param deviceSignal every device-related event will be sent here: find, connect, disconnect
    [[nodiscard]] virtual bool initialize( std::function<void(const std::string&)> deviceSignal = {} ) = 0;

    /// handle device state and call Viewer signals
    virtual void handle() = 0;
};

} //namespace MR
