#pragma once
#ifndef _WIN32
#ifndef __EMSCRIPTEN__
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
    typedef short unsigned int VendorId;
    typedef short unsigned int ProductId;
public:
    SpaceMouseHandlerHidapi();
    ~SpaceMouseHandlerHidapi();

    virtual void initialize() override;
    virtual void handle() override;

private:
    void startUpdateThread_();

    float convertCoord_( int coord_byte_low, int coord_byte_high );
    void convertInput_( const DataPacketRaw& packet, int packet_length, Vector3f& translate, Vector3f& rotate );

    bool findAndAttachDevice_();
    void printDevices_( struct hid_device_info *cur_dev );

private:
    hid_device *device_;
    std::thread updateThread_;
    bool updateThreadActive_;
    bool hasMousePackets_;
    std::mutex mtx_;
    std::condition_variable cv_;
    Vector3f translate_;
    Vector3f rotate_;

    const std::unordered_map<VendorId, std::vector<ProductId>> vendor2device_ = {
            { 0x046d, { 0xc603,    // spacemouse plus XT
                              0xc605,    // cadman
                              0xc606,    // spacemouse classic
                              0xc621,    // spaceball 5000
                              0xc623,    // space traveller
                              0xc625,    // space pilot
                              0xc626,    // space navigator
                              0xc627,    // space explorer
                              0xc628,    // space navigator for notebooks
                              0xc629,    // space pilot pro
                              0xc62b,    // space mouse pro
                              0xc640     // nulooq
                            }},
            { 0x256f, { 0xc62e,    // spacemouse wireless (USB cable)
                              0xc62f,    // spacemouse wireless receiver
                              0xc631,    // spacemouse pro wireless
                              0xc632,    // spacemouse pro wireless receiver
                              0xc633,    // spacemouse enterprise
                              0xc635,    // spacemouse compact
                              0xc652     // 3Dconnexion universal receiver
                            }}
    };
};

}
#endif
#endif