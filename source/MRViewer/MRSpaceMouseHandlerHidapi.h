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

class MRVIEWER_CLASS HandlerHidapi : public Handler, public PostFocusListener
{
public:
    HandlerHidapi();
    ~HandlerHidapi() override;

    bool initialize() override;
    void handle() override;

    /// returns true if this handler has connected device
    MRVIEWER_API bool hasValidDeviceConnected() const;

private:
    void initListenerThread_();
    virtual void postFocus_( bool focused ) override;

    void processAction_( const Action& action );

    bool findAndAttachDevice_( bool verbose );

private:
    hid_device* device_ = nullptr;

    class AtomicDevice
    {
    public:
        void resetDevice( VendorId vId = 0, ProductId pId = 0 );

        // update (rewrite its data) SpaceMouseAction if DataPacketRaw is not empty
        void parseRaw( const DataPacketRaw& packet, int packet_length, Action& action ) const;

        void process( const Action& action );
        bool valid() const;
    private:
        mutable std::mutex mutex_;
        std::unique_ptr<Device> device_;
    } smDevice_;

    size_t numMsg_ = 0;
    std::thread listenerThread_;
    std::atomic_bool terminateListenerThread_{ false };
    std::mutex syncThreadMutex_; // which thread reads and handles SpaceMouse data
    std::condition_variable cv_; // notify on thread change
    DataPacketRaw dataPacket_;    // packet from listener thread
    int packetLength_ = 0;
    std::atomic_bool active_{ false };
};

}
#endif
