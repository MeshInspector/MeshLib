#pragma once
#include "MRMesh/MRVector3.h"
#include <bitset>
#include <array>
#include <vector>
#include <unordered_map>
#include <string>

namespace MR::SpaceMouse
{

using VendorId = short unsigned int;
using ProductId = short unsigned int;
using DataPacketRaw = std::array<unsigned char, 13>;

/// enumeration all spacemouse buttons
enum SpaceMouseButtons : int
{
    SMB_NO = -1,
    SMB_MENU,

    SMB_ESC,
    SMB_ENTER,
    SMB_TAB,
    SMB_SHIFT,
    SMB_CTRL,
    SMB_ALT,
    SMB_SPACE,
    SMB_DELETE,

    SMB_CUSTOM_1,
    SMB_CUSTOM_2,
    SMB_CUSTOM_3,
    SMB_CUSTOM_4,
    SMB_CUSTOM_5,
    SMB_CUSTOM_6,
    SMB_CUSTOM_7,
    SMB_CUSTOM_8,
    SMB_CUSTOM_9,
    SMB_CUSTOM_10,
    SMB_CUSTOM_11,
    SMB_CUSTOM_12,

    SMB_FIT,
    SMB_TOP,
    SMB_RIGHT,
    SMB_FRONT,
    SMB_ROLL_CW, // roll clockwise
    SMB_LOCK_ROT,

    SMB_BTN_V1,
    SMB_BTN_V2,
    SMB_BTN_V3,
    SMB_ISO1,

    SMB_BUTTON_COUNT
};

/// known devices list
/// if you change this value, do not forget to update MeshLib/scripts/70-space-mouse-meshlib.rules
const std::unordered_map<VendorId, std::vector<ProductId>> cVendor2Device = {
        { VendorId( 0x046d ), {  // Logitech (3Dconnexion was a subsidiary)
                    0xc603,    // SpaceMouse plus XT
                    0xc605,    // cadman
                    0xc606,    // SpaceMouse classic
                    0xc621,    // spaceball 5000
                    0xc623,    // space traveller
                    0xc625,    // space pilot
                    0xc626,    // Full-size SpaceNavigator
                    0xc627,    // space explorer
                    0xc628,    // SpaceNavigator for notebooks
                    0xc629,    // space pilot pro
                    0xc62b,    // space mouse pro
                    0xc640     // nulooq
        }},
        { VendorId( 0x256f ), {  // 3Dconnexion
                    0xc62e,    // SpaceMouse wireless (USB cable)
                    0xc62f,    // SpaceMouse wireless receiver
                    0xc631,    // SpaceMouse pro wireless (USB cable)
                    0xc632,    // SpaceMouse pro wireless receiver
                    0xc633,    // SpaceMouse enterprise
                    0xc635,    // SpaceMouse compact
                    0xc638,    // SpaceMouse Pro Wireless Bluetooth Edition (USB cable)
                    0xc63a,    // SpaceMouse Wireless BT
                    0xc652,    // Universal receiver
                    0xc658     // Wireless (3DConnexion Universal Wireless Receiver in WIN32)
        }}
};


struct SpaceMouseAction {
    bool btnStateChanged = false;
    std::bitset<SMB_BUTTON_COUNT> buttons = 0;
    Vector3f translate = { 0.0f, 0.0f, 0.0f };
    Vector3f rotate = { 0.0f, 0.0f, 0.0f };
};

/// This class holds information and state of single SpaceMouse device
class SpaceMouseDevice
{
public:
    /// Updates internal data by device ids (does nothing if ids is same for current device)
    void updateDevice( VendorId vId, ProductId pId );
    
    /// Invalidates this device and its state
    void resetDevice();

    /// Check if this device is set and valid
    bool valid() const;

    /// Parses data from raw packet to unified `action`
    void parseRawEvent( const DataPacketRaw& raw, int numBytes, SpaceMouseAction& action ) const;

    /// Emit unified spacemouse signals in Viewer based on new action and current device state
    /// updates btn state if needed
    void processAction( const SpaceMouseAction& action );
private:
    VendorId vId_{ 0 };
    ProductId pId_{ 0 };
    std::bitset<SMB_BUTTON_COUNT> buttonsState_;

    const std::vector<std::vector<SpaceMouseButtons>>* buttonsMapPtr_ = nullptr;

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
