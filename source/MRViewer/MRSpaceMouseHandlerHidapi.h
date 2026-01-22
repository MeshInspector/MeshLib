#pragma once
#ifndef __EMSCRIPTEN__
#include "MRSpaceMouseHandler.h"
#include "MRSpaceMouseDevice.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRVector.h"

#include <hidapi/hidapi.h>

#include <atomic>
#include <bitset>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>

#ifdef __APPLE__
#include <hidapi/hidapi_darwin.h>
#endif
#ifdef _WIN32
#include "MRPch/MRWinapi.h"
#include <hidapi/hidapi_winapi.h>
#endif

namespace MR::SpaceMouse
{

class MRVIEWER_CLASS SpaceMouseHandlerHidapi : public SpaceMouseHandler, public PostFocusListener
{
public:
    SpaceMouseHandlerHidapi();
    ~SpaceMouseHandlerHidapi() override;

    bool initialize( std::function<void(const std::string&)> deviceSignal ) override;
    void handle() override;

    // set state of zoom by mouse scroll (to fix scroll signal from SpaceMouse driver)
    MRVIEWER_API void activateMouseScrollZoom( bool activeMouseScrollZoom );
    // get state of zoom by mouse scroll
    MRVIEWER_API bool isMouseScrollZoomActive()
    {
        return activeMouseScrollZoom_;
    }

private:
    void initListenerThread_();
    virtual void postFocus_( bool focused ) override;

    void processAction_( const SpaceMouseAction& action );

    // update (rewrite its data) SpaceMouseAction if DataPacketRaw is not empty
    void updateActionWithInput_( const DataPacketRaw& packet, int packet_length, SpaceMouseAction& action );

    bool findAndAttachDevice_( bool verbose );

private:
    std::function<void(const std::string&)> deviceSignal_;
    hid_device* device_ = nullptr;
    std::unique_ptr<SpaceMouseDevice> smDevice_;
    size_t numMsg_ = 0;
    std::thread listenerThread_;
    std::atomic_bool terminateListenerThread_{ false };
    std::mutex syncThreadMutex_; // which thread reads and handles SpaceMouse data
    std::condition_variable cv_; // notify on thread change
    DataPacketRaw dataPacket_;    // packet from listener thread
    int packetLength_ = 0;
    std::atomic_bool active_{ false };
    bool activeMouseScrollZoom_ = false;
};

}
#endif
