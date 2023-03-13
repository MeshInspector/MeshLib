#pragma once
#ifndef _WIN32
#include "MRSpaceMouseHandler.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRVector.h"
#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRMeshFwd.h"
#include <hidapi/hidapi.h>
#include <thread>
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#ifdef __APPLE__
#include <hidapi/hidapi_darwin.h>
#endif


namespace MR
{
class SpaceMouseHandlerHidapi : public SpaceMouseHandler
{
     typedef std::array<unsigned char, 13> DataPacketRaw;
public:
    SpaceMouseHandlerHidapi();
    ~SpaceMouseHandlerHidapi();

    virtual void initialize() override;
    virtual void handle() override;

private:
    void startUpdateThread_();

    float convertCoord( int coord_byte_low, int coord_byte_high );
    void convertInput( const DataPacketRaw& packet, int packet_length, Vector3f& translate, Vector3f& rotate );

    struct HidDevice;
    void findDevice( struct hid_device_info *cur_dev, HidDevice& attachDevice );
    void printDevices( struct hid_device_info *cur_dev );

private:
    struct HidDevice {
        HidDevice() = default;
        HidDevice( short unsigned int vendor_id, short unsigned int product_id)
            : vendor_id_(vendor_id), product_id_(product_id) {}
        short unsigned int vendor_id_;
        short unsigned int product_id_;
    };
    hid_device *device_;
    std::thread updateThread_;
    bool updateThreadActive_;
    bool hasMousePackets_;
    std::mutex mtx_;
    std::condition_variable cv_;
    Vector3f translate_;
    Vector3f rotate_;

    const std::vector<HidDevice> supportedDevices_ =
    {
            {0x046d, 0xc603},	// spacemouse plus XT
            {0x046d, 0xc605},	// cadman
            {0x046d, 0xc606},	// spacemouse classic
            {0x046d, 0xc621},	// spaceball 5000
            {0x046d, 0xc623},	// space traveller
            {0x046d, 0xc625},	// space pilot
            {0x046d, 0xc626},	// space navigator
            {0x046d, 0xc627},	// space explorer
            {0x046d, 0xc628},	// space navigator for notebooks
            {0x046d, 0xc629},	// space pilot pro
            {0x046d, 0xc62b},	// space mouse pro
            {0x046d, 0xc640},	// nulooq
            {0x256f, 0xc62e},	// spacemouse wireless (USB cable)
            {0x256f, 0xc652},	// spacemouse wireless receiver
            {0x256f, 0xc631},	// spacemouse pro wireless
            {0x256f, 0xc632},	// spacemouse pro wireless receiver
            {0x256f, 0xc633},	// spacemouse enterprise
            {0x256f, 0xc635},	// spacemouse compact
            {0x256f, 0xc636},	// spacemouse module
            {0x256f, 0xc652}	    // 3Dconnexion universal receiver
    };
};

}
#endif