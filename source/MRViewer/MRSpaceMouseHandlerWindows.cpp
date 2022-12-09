#ifdef _WIN32
#include "MRSpaceMouseHandlerWindows.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include <windows.h>
#include <GLFW/glfw3.h>
#include <functional>

namespace MR
{

constexpr float axesScale = 93.62f; // experemental coefficient to scale raw axes data to range [-1 ; 1]

constexpr int mapButtonsCompact[2] = {
    SMB_CUSTOM_1, SMB_CUSTOM_2
};
constexpr int mapButtonsPro[15] = {
    SMB_MENU, SMB_FIT,
    SMB_TOP, SMB_RIGHT, SMB_FRONT, SMB_ROLL_CW,
    SMB_CUSTOM_1, SMB_CUSTOM_2, SMB_CUSTOM_3, SMB_CUSTOM_4,
    SMB_ESC, SMB_ALT, SMB_SHIFT, SMB_CTRL,
    SMB_LOCK_ROT
};
// TODO !!! NOT TESTED !!!
constexpr int mapButtonsEnterprise[31] = {
    SMB_MENU, SMB_FIT,
    SMB_TOP, SMB_RIGHT, SMB_FRONT, SMB_ROLL_CW, SMB_LOCK_ROT,
    SMB_ISO1, SMB_BTN_V1, SMB_BTN_V2, SMB_BTN_V3,
    SMB_CUSTOM_1, SMB_CUSTOM_2, SMB_CUSTOM_3, SMB_CUSTOM_4, SMB_CUSTOM_5, SMB_CUSTOM_6,
    SMB_CUSTOM_7, SMB_CUSTOM_8, SMB_CUSTOM_9, SMB_CUSTOM_10, SMB_CUSTOM_11, SMB_CUSTOM_12,
    SMB_ESC, SMB_ENTER, SMB_ALT, SMB_SHIFT, SMB_CTRL, SMB_TAB, SMB_SPACE, SMB_DELETE
};

constexpr DWORD logitechId = 0x46d;
constexpr DWORD connexionId = 0x256f;

// Array of input device examples to register
RAWINPUTDEVICE inputDevices[] = {
    {0x01, 0x08, 0x00, 0x00}, // Usage Page = 0x01 Generic Desktop Page, Usage Id = 0x08 Multi-axis Controller
    {0x01, 0x05, 0x00, 0x00}, // Game Pad
    {0x01, 0x04, 0x00, 0x00} // Joystick
};
constexpr int inputDevicesCount = sizeof( inputDevices ) / sizeof( inputDevices[0] );

bool isSpaceMouseAttached()
{
    unsigned int devicesCount = 0;

    if ( GetRawInputDeviceList( NULL, &devicesCount, sizeof( RAWINPUTDEVICELIST ) ) != 0 )
        return false;

    if ( devicesCount == 0 )
        return false;

    std::vector<RAWINPUTDEVICELIST> rawInputDeviceList( devicesCount );
    if ( GetRawInputDeviceList( rawInputDeviceList.data(), &devicesCount, sizeof( RAWINPUTDEVICELIST ) ) == unsigned int( -1 ) )
        return false;

    for ( unsigned int i = 0; i < devicesCount; ++i )
    {
        RID_DEVICE_INFO rdi = { sizeof( rdi ) };
        unsigned int cbSize = sizeof( rdi );

        if ( GetRawInputDeviceInfo( rawInputDeviceList[i].hDevice, RIDI_DEVICEINFO, &rdi, &cbSize ) > 0 )
        {
            //skip non HID and non logitec (3DConnexion) devices
            if ( !( rdi.dwType == RIM_TYPEHID && ( rdi.hid.dwVendorId == logitechId || rdi.hid.dwVendorId == connexionId ) ) )
                continue;

            //check if devices matches Multi-axis Controller
            for ( unsigned int j = 0; j < inputDevicesCount; ++j )
            {
                if ( inputDevices[j].usUsage == rdi.hid.usUsage && inputDevices[j].usUsagePage == rdi.hid.usUsagePage )
                {
                    return true;
                }
            }
        }
    }
    return false;
}

bool InitializeRawInput()
{
    unsigned int cbSize = sizeof( inputDevices[0] );
    for ( size_t i = 0; i < inputDevicesCount; i++ )
    {
        // Set the target window to use
        //inputDevices[i].hwndTarget = hwndTarget;

        // enable receiving the WM_INPUT_DEVICE_CHANGE message.
        inputDevices[i].dwFlags |= RIDEV_DEVNOTIFY;
    }
    return ( RegisterRawInputDevices( inputDevices, inputDevicesCount, cbSize ) != FALSE );
}

SpaceMouseHandlerWindows::SpaceMouseHandlerWindows()
{
    std::for_each( buttons_.begin(), buttons_.end(), [](auto& v) { v = 0; } );
    std::for_each( axesOld_.begin(), axesOld_.end(), [](auto& v) { v = 0; } );
    axesDiff_ = { 0, 0, 0, 0, 0, 0 };
    connect( &getViewerInstance() );
}

SpaceMouseHandlerWindows::~SpaceMouseHandlerWindows()
{
    if ( updateThread_.joinable() )
        updateThread_.join();
}

void SpaceMouseHandlerWindows::initialize()
{
    bool spaceMouseAttached = isSpaceMouseAttached();
    if ( spaceMouseAttached )
        spdlog::info( "Found attached SpaceMouse" );
    else
    {
        spdlog::info( "Not found any attached SpaceMouse" );
        return;
    }

    initialized_ = InitializeRawInput();
    spdlog::info( "Initialize SpaceMouse {}", initialized_ ? "success" : "failed" );

    updateConnected_();
}

void SpaceMouseHandlerWindows::handle()
{
    if ( !initialized_ || joystickIndex_ == -1 || !active_ )
        return;

    auto& viewer = getViewerInstance();
    int count;
	std::array<float, 6> axesDiff = axesDiff_;
    if ( std::any_of( axesDiff.begin(), axesDiff.end(), [] ( const float& v ) { return std::fabs( v ) > 1.e-2 / axesScale; } ) )
    {
        Vector3f translate( axesDiff[0], axesDiff[1], axesDiff[2] );
        Vector3f rotate( axesDiff[3], axesDiff[4], axesDiff[5] );
        translate *= axesScale;
        rotate *= axesScale;
        
		viewer.spaceMouseMove( translate, rotate );
		axesDiff_ = { 0, 0, 0, 0, 0, 0 };
    }

    const unsigned char* buttons = glfwGetJoystickButtons( joystickIndex_, &count );
    // SpaceMouse Compact have 2 btns, Pro - 15, Enterprise - 31
    if ( count == 2 || count == 15 || count == 31 )
    {
        buttonsCount_ = count;
        if ( active_ )
        {
            mapButtons_ = mapButtonsPro;
            if ( count == 2 )
                mapButtons_ = mapButtonsCompact;
            else if ( count == 15 )
                mapButtons_ = mapButtonsPro;
            else
                mapButtons_ = mapButtonsEnterprise;
            for ( int i = 0; i < count; ++i )
            {
                int button = mapButtons_[i];
                if ( !buttons_[i] && buttons[i] ) // button down
                    viewer.spaceMouseDown( button );
                else if ( buttons_[i] && !buttons[i] ) // button up
                    viewer.spaceMouseUp( button );
//                 else if ( buttons_[i] && buttons[i] &&  ) // button repeat
//                     viewer.spaceMouseRepeat( button );
            }
        }
        std::copy( buttons, buttons + count, buttons_.begin() );
    }
}

void SpaceMouseHandlerWindows::updateConnected( int /*jid*/, int /*event*/ )
{
    if ( initialized_ )
        updateConnected_();
    else
        initialize();
}

void SpaceMouseHandlerWindows::postFocusSignal_( bool focused )
{
    active_ = focused;
}

void SpaceMouseHandlerWindows::updateConnected_()
{

    int newJoystickIndex = -1;
    for ( int i = GLFW_JOYSTICK_1; i <= GLFW_JOYSTICK_LAST; ++i )
    {
        int present = glfwJoystickPresent( i );
        if ( !present )
            continue;
        const char* name = glfwGetJoystickName( i );
        std::string_view str( name );
        auto findRes = str.find( "SpaceMouse" );
        if ( findRes != std::string_view::npos )
        {
            newJoystickIndex = i;
            break;
        }
    }

    if ( newJoystickIndex == joystickIndex_ )
        return;

    auto& viewer = getViewerInstance();
    if ( joystickIndex_ != -1 )
    {
        for ( int i = 0; i < buttonsCount_; ++i )
        {
            if ( buttons_[i] ) // button up
                viewer.spaceMouseUp( mapButtons_[i] );
		}
        std::for_each( buttons_.begin(), buttons_.end(), [](auto& v) { v = 0; } );

        if ( newJoystickIndex == -1 && updateThread_.joinable() )
            updateThread_.join();
    }

    if ( newJoystickIndex != -1 )
    {
        int count;
        const float* axesNew = glfwGetJoystickAxes( newJoystickIndex, &count );
        std::copy( axesNew, axesNew + 6, axesOld_.begin() );

        if ( joystickIndex_ == -1 )
            startUpdateThread_();
    }

	joystickIndex_ = newJoystickIndex;
    getViewerInstance().mouseController.setMouseScroll( joystickIndex_ == -1 );
}

void SpaceMouseHandlerWindows::startUpdateThread_()
{
    updateThreadActive_ = true;
    axesDiff_ = { 0, 0, 0, 0, 0, 0 };
	updateThread_ = std::thread( [&]
    {
        std::array<float, 6> axesDiff;
        std::for_each( axesDiff.begin(), axesDiff.end(), [](float& v) { v = 0.f; } );
       int count = 0;
       do
       {
           if ( active_ && joystickIndex_ != -1 )
		   {
               const float* axesNew = glfwGetJoystickAxes( joystickIndex_, &count );
               if ( count == 6 )
			   {
				   axesDiff = axesDiff_;
                   for ( int i = 0; i < 6; ++i )
                   {
                       float newDiff = axesNew[i] - axesOld_[i];
                       if ( newDiff * axesDiff[i] < 0 )
                           axesDiff[i] = newDiff;
                       else if ( std::fabs( axesDiff[i] ) < std::fabs( newDiff ) )
                           axesDiff[i] = newDiff;
                   }
                   axesDiff_ = axesDiff;
                   std::copy( axesNew, axesNew + 6, axesOld_.begin() );
               }
           }
           std::this_thread::sleep_for( std::chrono::microseconds( 100 ) );
       } while ( updateThreadActive_ );
    } );
}

}

#endif
