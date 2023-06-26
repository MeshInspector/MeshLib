#pragma once
#ifndef __EMSCRIPTEN__
#include "MRSpaceMouseHandler.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRVector.h"
#include "MRPch/MRSpdlog.h"
#include <hidapi/hidapi.h>
#include <bitset>
#include <condition_variable>
#include <mutex>
#include <thread>
#ifdef __APPLE__
#include <hidapi/hidapi_darwin.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <hidapi/hidapi_winapi.h>
#endif

namespace MR
{
class MRVIEWER_CLASS SpaceMouseHandlerHidapi : public SpaceMouseHandler, public PostFocusListener
{
    typedef std::array<unsigned char, 13> DataPacketRaw;
    typedef short unsigned int VendorId;
    typedef short unsigned int ProductId;
    struct SpaceMouseAction {
        bool isButtonStateChanged = false;
        std::bitset<SMB_BUTTON_COUNT> buttons = 0;
        Vector3f translate = { 0.0f, 0.0f, 0.0f };
        Vector3f rotate = { 0.0f, 0.0f, 0.0f };
    };
public:
    SpaceMouseHandlerHidapi();
    ~SpaceMouseHandlerHidapi();

    virtual void initialize() override;
    virtual void handle() override;

    // set state of zoom by mouse scroll (to fix scroll signal from spacemouse driver)
    MRVIEWER_API void activateMouseScrollZoom( bool activeMouseScrollZoom );
    // get state of zoom by mouse scroll
    MRVIEWER_API bool isMouseScrollZoomActive()
    {
        return activeMouseScrollZoom_;
    }

private:
    void initListenerThread_();
    void setButtonsMap_( VendorId vendorId, ProductId productId );
    virtual void postFocus_( bool focused ) override;

    void processAction_( const SpaceMouseAction& action );
    float convertCoord_( int coord_byte_low, int coord_byte_high );

    // update (rewrite its data) SpaceMouseAction if DataPacketRaw is not empty
    void updateActionWithInput_( const DataPacketRaw& packet, int packet_length, SpaceMouseAction& action );

    bool findAndAttachDevice_();
    void printDevices_( struct hid_device_info* cur_dev );

private:
    hid_device* device_;
    const std::vector<std::vector<SpaceMouseButtons>>* buttonsMapPtr_;
    std::bitset<SMB_BUTTON_COUNT> buttonsState_;
    std::thread listenerThread_;
    std::atomic_bool terminateListenerThread_;
    std::mutex syncThreadMutex_; // which thread reads and handles SpaceMouse data
    std::condition_variable cv_; // notify on thread change
    DataPacketRaw dataPacket_;    // packet from listener thread
    int packetLength_;
    std::atomic_bool active_;
    bool activeMouseScrollZoom_;

    // if you change this value, do not forget to update MeshLib/scripts/70-space-mouse-meshlib.rules
    const std::unordered_map<VendorId, std::vector<ProductId>> vendor2device_ = {
            { VendorId(0x046d), { 0xc603,    // spacemouse plus XT
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
            { VendorId(0x256f), { 0xc62e,    // spacemouse wireless (USB cable)
                        0xc62f,    // spacemouse wireless receiver
                        0xc631,    // spacemouse pro wireless
                        0xc632,    // spacemouse pro wireless receiver
                        0xc633,    // spacemouse enterprise
                        0xc635,    // spacemouse compact
                        0xc652     // 3Dconnexion universal receiver
            }}
    };

    /*         |   <--- packet values --->
     *   #byte |   1          2
     *   ------+--------------------------
     *       0 |   -          -
     *       1 |   custom_1   custom_2
     */
    std::vector<std::vector<SpaceMouseButtons>> buttonMapCompact = {
        {  }, // 0th byte (unused)
        { SMB_CUSTOM_1, SMB_CUSTOM_2} // 1st byte
    };

    /*         |  <--- packet values --->
     *   #byte |    1      2       4       8     16      32      64      126
     *   ------+------------------------------------------------------------------
     *       1 |  menu   fit     T               R       F
     *       2 |  rot                            1       2       3        4
     *       3 |                                                 esc      alt
     *       4 |  shift  ctrl    lock
     *
     */
    std::vector<std::vector<SpaceMouseButtons>> buttonMapPro = {
        {  }, // 0th byte (unused)
        //1             2             4             8             16            32            64            128
        { SMB_MENU,     SMB_FIT,      SMB_TOP,      SMB_NO,       SMB_RIGHT,    SMB_FRONT,    SMB_NO,       SMB_NO },      // 1st byte
        { SMB_ROLL_CW,  SMB_NO,       SMB_NO,       SMB_NO,       SMB_CUSTOM_1, SMB_CUSTOM_2, SMB_CUSTOM_3, SMB_CUSTOM_4}, // 2nd byte
        { SMB_NO,       SMB_NO,       SMB_NO,       SMB_NO,       SMB_NO,       SMB_NO,       SMB_ESC,      SMB_ALT},      // 3rd byte
        { SMB_SHIFT,    SMB_CTRL,     SMB_LOCK_ROT, SMB_NO,       SMB_NO,       SMB_NO,       SMB_NO,       SMB_NO,},      // 4th byte
    };

    // @TODO !!! NOT TESTED !!!
    std::vector<std::vector<SpaceMouseButtons>> buttonMapEnterprise = {
        {  }, // 0th byte (unused)
        { SMB_CUSTOM_1,     SMB_CUSTOM_2}
    };
    /* @TODO !!! NOT TESTED !!!
    static constexpr int mapButtonsEnterprise[31] = {
        SMB_MENU, SMB_FIT,
        SMB_TOP, SMB_RIGHT, SMB_FRONT, SMB_ROLL_CW, SMB_LOCK_ROT,
        SMB_ISO1, SMB_BTN_V1, SMB_BTN_V2, SMB_BTN_V3,
        SMB_CUSTOM_1, SMB_CUSTOM_2, SMB_CUSTOM_3, SMB_CUSTOM_4, SMB_CUSTOM_5, SMB_CUSTOM_6,
        SMB_CUSTOM_7, SMB_CUSTOM_8, SMB_CUSTOM_9, SMB_CUSTOM_10, SMB_CUSTOM_11, SMB_CUSTOM_12,
        SMB_ESC, SMB_ENTER, SMB_ALT, SMB_SHIFT, SMB_CTRL, SMB_TAB, SMB_SPACE, SMB_DELETE
    };
    */
};

}
#endif
