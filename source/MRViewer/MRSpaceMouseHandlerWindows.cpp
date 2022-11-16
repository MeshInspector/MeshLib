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

#define LOGITECH_VENDOR_ID 0x46d
#define CONNEXION_VENDOR_ID  0x256f

PRAWINPUTDEVICE GetDevicesToRegister( unsigned int* pNumDevices )
{
    // Array of raw input devices to register
    static RAWINPUTDEVICE sRawInputDevices[] = {
        {0x01, 0x08, 0x00, 0x00} // Usage Page = 0x01 Generic Desktop Page, Usage Id= 0x08 Multi-axis Controller
       ,{0x01, 0x05, 0x00, 0x00} // game pad
       ,{0x01, 0x04, 0x00, 0x00} // joystick
    };

    if ( pNumDevices )
    {
        *pNumDevices = sizeof( sRawInputDevices ) / sizeof( sRawInputDevices[0] );
    }

    return sRawInputDevices;
}

bool Is3dmouseAttached()
{
    unsigned int numDevicesOfInterest = 0;
    PRAWINPUTDEVICE devicesToRegister = GetDevicesToRegister( &numDevicesOfInterest );

    unsigned int nDevices = 0;

    if ( ::GetRawInputDeviceList( NULL, &nDevices, sizeof( RAWINPUTDEVICELIST ) ) != 0 )
        return false;

    if ( nDevices == 0 )
        return false;

    std::vector<RAWINPUTDEVICELIST> rawInputDeviceList( nDevices );
    if ( ::GetRawInputDeviceList( &rawInputDeviceList[0], &nDevices, sizeof( RAWINPUTDEVICELIST ) ) == static_cast< unsigned int >( -1 ) )
        return false;

    for ( unsigned int i = 0; i < nDevices; ++i )
    {
        RID_DEVICE_INFO rdi = { sizeof( rdi ) };
        unsigned int cbSize = sizeof( rdi );

        if ( GetRawInputDeviceInfo( rawInputDeviceList[i].hDevice, RIDI_DEVICEINFO, &rdi, &cbSize ) > 0 )
        {
            //skip non HID and non logitec (3DConnexion) devices
            if ( !( rdi.dwType == RIM_TYPEHID && ( rdi.hid.dwVendorId == LOGITECH_VENDOR_ID || rdi.hid.dwVendorId == CONNEXION_VENDOR_ID ) ) )
                continue;

            //check if devices matches Multi-axis Controller
            for ( unsigned int j = 0; j < numDevicesOfInterest; ++j )
            {
                if ( devicesToRegister[j].usUsage == rdi.hid.usUsage && devicesToRegister[j].usUsagePage == rdi.hid.usUsagePage )
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
    unsigned int numDevices = 0;
    PRAWINPUTDEVICE devicesToRegister = GetDevicesToRegister( &numDevices );

    if ( numDevices == 0 )
        return false;

    // Get OS version.
    //OSVERSIONINFO osvi = { sizeof(OSVERSIONINFO), 0 };
    //::GetVersionEx(&osvi);

    unsigned int cbSize = sizeof( devicesToRegister[0] );
    for ( size_t i = 0; i < numDevices; i++ )
    {
        // Set the target window to use
        //devicesToRegister[i].hwndTarget = hwndTarget;

        // If Vista or newer, enable receiving the WM_INPUT_DEVICE_CHANGE message.
        //if (osvi.dwMajorVersion >= 6) {
        devicesToRegister[i].dwFlags |= RIDEV_DEVNOTIFY;
    }
    return ( ::RegisterRawInputDevices( devicesToRegister, numDevices, cbSize ) != FALSE );
}

void SpaceMouseHandlerWindows::initialize()
{
    bool is3Dmouse = Is3dmouseAttached();
    spdlog::info( "Is3dmouseAttached = {}", is3Dmouse );

    if ( is3Dmouse )
    {
        initialized_ = InitializeRawInput();
        spdlog::info( "InitializeRawInput = {}", initialized_ );
    }

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
    std::memcpy( axes_.data(), axesNew, sizeof( float ) * 6 );
    translate *= axesScale;
    rotate *= axesScale;

    auto& viewer = getViewerInstance();
    if ( translate.lengthSq() > 1.e-3f || rotate.lengthSq() > 1.e-3f )
        viewer.spaceMouseMove( translate, rotate );


    const unsigned char* buttons = glfwGetJoystickButtons( joystickIndex_, &count );
    for ( int i = 0; i < BUTTON_COUNT; ++i )
    {
        if ( !buttons_[i] && buttons[i] ) // button down
            viewer.spaceMouseDown( i );
        else if ( buttons_[i] && !buttons[i] ) // button up
            viewer.spaceMouseUp( i );
//         else if ( buttons_[i] && buttons[i] &&  ) // button repeat
//             viewer.spaceMouseRepeat( i );
    }
    std::memcpy( buttons_.data(), buttons, sizeof( unsigned char ) * count );
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
        std::memcpy( axes_.data(), axesNew, sizeof( float ) * 6 );
    }
}

}
