#ifdef _WIN32
#include "MRSpaceMouseHandlerWindows.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include <windows.h>
#include <GLFW/glfw3.h>
#include <functional>

namespace MR
{

constexpr float axesScale = 93.62f; // experemental coefficient to scale raw axes data to range [-1 ; 1]

//constexpr mapButtons2 = {};
//constexpr mapButtons15 = {};
//constexpr mapButtons31 = {};

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

void SpaceMouseHandlerWindows::initialize()
{
    bool spaceMouseAttached = isSpaceMouseAttached();
    if ( spaceMouseAttached )
        spdlog::info( "Found attached spacemouse" );
    else
    {
        spdlog::info( "Not found any attached spacemouse" );
        return;
    }

    initialized_ = InitializeRawInput();
    spdlog::info( "InitializeRawInput = {}", initialized_ );

    updateConnected_();
}

void SpaceMouseHandlerWindows::handle()
{
    if ( !initialized_ || joystickIndex_ == -1 )
        return;

    int count;
    const float* axesNew = glfwGetJoystickAxes( joystickIndex_, &count );
    if ( count != 6 )
    {
        spdlog::error( "Error SpaceMouseHandlerWindows : Wrong axes count" );
        assert( false );
    }
    Vector3f translate( axesNew[0] - axes_[0], axesNew[1] - axes_[1], axesNew[2] - axes_[2] );
    Vector3f rotate( axesNew[3] - axes_[3], axesNew[4] - axes_[4], axesNew[5] - axes_[5] );
    std::copy( axesNew, axesNew + 6, axes_.begin() );
    translate *= axesScale;
    rotate *= axesScale;

    auto& viewer = getViewerInstance();
    if ( translate.lengthSq() > 1.e-3f || rotate.lengthSq() > 1.e-3f )
        viewer.spaceMouseMove( translate, rotate );


    const unsigned char* buttons = glfwGetJoystickButtons( joystickIndex_, &count );
    for ( int i = 0; i < count; ++i )
    {
        if ( !buttons_[i] && buttons[i] ) // button down
            viewer.spaceMouseDown( i );
        else if ( buttons_[i] && !buttons[i] ) // button up
            viewer.spaceMouseUp( i );
//         else if ( buttons_[i] && buttons[i] &&  ) // button repeat
//             viewer.spaceMouseRepeat( i );
    }
    std::copy( buttons, buttons + count, buttons_.begin() );
}

void SpaceMouseHandlerWindows::updateConnected( int /*jid*/, int /*event*/ )
{
    updateConnected_();
}

void SpaceMouseHandlerWindows::updateConnected_()
{
    joystickIndex_ = -1;
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
            joystickIndex_ = i;
            break;
        }
    }

    if ( joystickIndex_ != -1 )
    {
        int count;
        const float* axesNew = glfwGetJoystickAxes( joystickIndex_, &count );
        std::copy( axesNew, axesNew + 6, axes_.begin() );
    }
}

}

#endif
