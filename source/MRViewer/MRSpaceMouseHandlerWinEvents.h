#pragma once
#ifdef _WIN32
#include "MRViewerFwd.h"
#include "MRSpaceMouseHandler.h"
#include "MRSpaceMouseDevice.h"
#include "MRMesh/MRSignal.h"

namespace MR
{
class Win32MessageHandler;
}

namespace MR::SpaceMouse
{

class MRVIEWER_CLASS HandlerWinEvents : public Handler
{
public:

    bool initialize() override;

    // there is no need for this function in WinEvent handler
    virtual void handle() override {}

    /// returns true if this handler has connected device
    MRVIEWER_API bool hasValidDeviceConnected() const;
private:
    std::unique_ptr<Device> device_;

    std::shared_ptr<Win32MessageHandler> winHandler_;
    boost::signals2::scoped_connection winEventsConnection_;
    size_t numMsg_ = 0;

    void resetDevice_( void* handle );
};

}
#endif // _WIN32
