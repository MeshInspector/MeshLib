#include "MRSpaceMouseHandlerWindows.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include <windows.h>
#include <GLFW/glfw3.h>

namespace MR
{

static PRAWINPUTDEVICE GetDevicesToRegister( unsigned int* pNumDevices )
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

#define LOGITECH_VENDOR_ID 0x46d
#define CONNEXION_VENDOR_ID  0x256f

bool Is3dmouseAttached()
{
    unsigned int numDevicesOfInterest = 0;
    PRAWINPUTDEVICE devicesToRegister = GetDevicesToRegister( &numDevicesOfInterest );

    unsigned int nDevices = 0;

    if ( ::GetRawInputDeviceList( NULL, &nDevices, sizeof( RAWINPUTDEVICELIST ) ) != 0 )
    {
        return false;
    }

    if ( nDevices == 0 )
        return false;

    std::vector<RAWINPUTDEVICELIST> rawInputDeviceList( nDevices );
    if ( ::GetRawInputDeviceList( &rawInputDeviceList[0], &nDevices, sizeof( RAWINPUTDEVICELIST ) )
        == static_cast< unsigned int >( -1 ) )
    {
        return false;
    }

    for ( unsigned int i = 0; i < nDevices; ++i )
    {
        RID_DEVICE_INFO rdi = { sizeof( rdi ) };
        unsigned int cbSize = sizeof( rdi );

        if ( GetRawInputDeviceInfo( rawInputDeviceList[i].hDevice, RIDI_DEVICEINFO, &rdi, &cbSize ) > 0 )
        {
            //skip non HID and non logitec (3DConnexion) devices
            if ( !( rdi.dwType == RIM_TYPEHID
                && ( rdi.hid.dwVendorId == LOGITECH_VENDOR_ID
                    || rdi.hid.dwVendorId == CONNEXION_VENDOR_ID ) ) )
            {
                continue;
            }

            //check if devices matches Multi-axis Controller
            for ( unsigned int j = 0; j < numDevicesOfInterest; ++j )
            {
                if ( devicesToRegister[j].usUsage == rdi.hid.usUsage
                    && devicesToRegister[j].usUsagePage == rdi.hid.usUsagePage )
                {
                    return true;
                }
            }
        }
    }
    return false;
}

static HWND fWindow;

bool InitializeRawInput( HWND hwndTarget )
{
    fWindow = hwndTarget;

    // Simply fail if there is no window
    if ( !hwndTarget )
        return false;

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
        //}
    }
    return ( ::RegisterRawInputDevices( devicesToRegister, numDevices, cbSize ) != FALSE );
}


SpaceMouseHandlerWindows::SpaceMouseHandlerWindows()    
{
}

void SpaceMouseHandlerWindows::initialize()
{
    bool is3Dmouse = Is3dmouseAttached();
    spdlog::info( "Is3dmouseAttached = {}", is3Dmouse );

    if ( is3Dmouse )
    {
        initialized_ = InitializeRawInput( GetConsoleWindow() );
        spdlog::info( "InitializeRawInput = {}", initialized_ );
    }
}

void SpaceMouseHandlerWindows::handle()
{
    if ( !initialized_ )
        return;

    int count;
    const float* axesNew = glfwGetJoystickAxes( GLFW_JOYSTICK_1, &count );
    if ( count != 6 )
    {
        spdlog::error( "Error SpaceMouseHandlerWindows : Wrong axes count" );
        assert( false );
    }
    Vector3f translate( axesNew[0] - axes_[0], axesNew[1] - axes_[1], axesNew[2] - axes_[2] );
    Vector3f rotate( axesNew[3] - axes_[3], axesNew[4] - axes_[4], axesNew[5] - axes_[5] );
    std::memcpy( axes_.data(), axesNew, sizeof( float ) * 6 );
    translate = mult( translate, translateScale_ );
    rotate = mult( rotate, rotateScale_ );
    if ( ( translate.lengthSq() > 1.e-3f  && translate.lengthSq() < 1.e+3f ) ||
        ( rotate.lengthSq() > 1.e-3f && rotate.lengthSq() < 1.e+3f ) )
        getViewerInstance().spaceMouseMove( translate, rotate );




    //         spdlog::info( "joystick present =====================================" );
    //         for ( int i = GLFW_JOYSTICK_1; i < GLFW_JOYSTICK_LAST; ++i )
    //         {
    //             int present = glfwJoystickPresent( i );
    //             spdlog::info( "present {} = {}", i, present );
    //         }

    //         const char* name = glfwGetJoystickName( GLFW_JOYSTICK_1 );
    //         spdlog::info( "name = {}", name );

//     int count;
//     const unsigned char* buttons = glfwGetJoystickButtons( GLFW_JOYSTICK_1, &count );
//     spdlog::info( "joystick buttons ===================================== time = {}", glfwGetTime() );
//     for ( int i = 0; i < count; ++i )
//     {
//         spdlog::info( "button {} = {}", i, int( buttons[i] ) );
//     }

//         spdlog::info( "joystick axes =====================================" );
//         for ( int i = 0; i < count; ++i )
//         {
//             spdlog::info( "axis {} = {}", i, axesNew[i] );
//         }
}

}
