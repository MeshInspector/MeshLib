#pragma once

#ifdef __APPLE__
#include "MRSpaceMouseHandler.h"
#include "MRViewerEventsListener.h"

namespace MR
{

/**
 * SpaceMouse handler using the official 3DxWare driver for macOS.
 * As the driver takes the exclusive control of the SpaceMouse devices, there is no way to connect to the devices
 * other than using the driver.
 */
class SpaceMouseHandler3dxMacDriver : public SpaceMouseHandler
{
public:
    SpaceMouseHandler3dxMacDriver();
    ~SpaceMouseHandler3dxMacDriver() override;

    void setClientName( const char* name, size_t len = 0 );

public:
    // SpaceMouseHandler
    bool initialize( std::function<void(const std::string&)> deviceSignal ) override;
    void handle() override;

private:
    std::unique_ptr<uint8_t[]> clientName_;
    uint16_t clientId_{ 0 };
};

} // namespace MR
#endif
