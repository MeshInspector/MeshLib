#include "MRSpaceMouseHandlerWinEvents.h"
#ifdef _WIN32
#include "MRViewer.h"
#include "MRMesh/MRTelemetry.h"
#include "MRWin32MessageHandler.h"
#include "MRMesh/MRStringConvert.h"

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include "MRPch/MRWinapi.h"

#pragma warning(push)
#pragma warning(disable: 4191)
#include <hidsdi.h>
#pragma warning(pop)
#include <hidusage.h>

namespace MR::SpaceMouse
{

bool HandlerWinEvents::initialize()
{
    auto window = getViewerInstance().window;
    if ( !window )
    {
        assert( !"no glfw window while trying to initialize SpaceMouse::HandlerWinEvents" );
        return false;
    }
    auto winHndl = glfwGetWin32Window( window );
    if ( !winHndl )
    {
        assert( !"no window handler while trying to initialize SpaceMouse::HandlerWinEvents" );
        return false;
    }

    std::array<RAWINPUTDEVICE, 1> rawDevices;
    for ( auto& rawDevice : rawDevices )
    {
        rawDevice.usUsagePage = HID_USAGE_PAGE_GENERIC;
        rawDevice.usUsage = HID_USAGE_GENERIC_MULTI_AXIS_CONTROLLER;
        rawDevice.hwndTarget = winHndl;
        rawDevice.dwFlags = 0;
    }
    if ( !RegisterRawInputDevices( rawDevices.data(), unsigned( rawDevices.size() ), sizeof( RAWINPUTDEVICE ) ) )
    {
        spdlog::warn( "Could not register raw devices" );
        return false;
    }

    auto handler = Win32MessageHandler::getHandler( winHndl );
    if ( !handler || !handler->isValid() )
    {
        assert( !"invalid Win32MessageHandler while trying to initialize SpaceMouse::HandlerWinEvents" );
        return false;
    }

    winHandler_ = std::move( handler );

    auto processRawInput = [this] ( HWND, UINT message, WPARAM /*wParam*/, LPARAM lParam )->bool
    {
        if ( message != WM_INPUT )
            return false;

        RAWINPUT rawInput;
        UINT rawSize = sizeof( RAWINPUT );
        GetRawInputData( ( HRAWINPUT )lParam, RID_INPUT, &rawInput, &rawSize, sizeof( RAWINPUTHEADER ) );
        if ( rawInput.header.dwType != RIM_TYPEHID )
            return false;

        resetDevice_( rawInput.header.hDevice );
        
        Action action;
        DataPacketRaw rawPacket;
        unsigned packetSize = rawInput.data.hid.dwSizeHid;
        std::copy( rawInput.data.hid.bRawData, rawInput.data.hid.bRawData + std::min( size_t( rawInput.data.hid.dwSizeHid ), rawPacket.size() ), rawPacket.data() );
        device_->parseRawEvent( rawPacket, packetSize, action );

        ++numMsg_;
        if ( numMsg_ == 1 )
            TelemetrySignal( "WIN API first action processing" );
        if ( std::popcount( numMsg_ ) == 1 ) // report every power of 2
            TelemetrySignal( "WIN API SpaceMouse next log messages" );
        device_->processAction( action );

        return false; // to pass this event further
    };

    winEventsConnection_ = winHandler_->onMessage.connect( processRawInput );
    return true;
}

bool HandlerWinEvents::hasValidDeviceConnected() const
{
    return device_ && device_->valid();
}

void HandlerWinEvents::resetDevice_( void* handle )
{
    RID_DEVICE_INFO info;
    unsigned infoSize = sizeof( RID_DEVICE_INFO );
    info.cbSize = infoSize;

    GetRawInputDeviceInfo( handle, RIDI_DEVICEINFO, &info, &infoSize );

    VendorId vId = VendorId( info.hid.dwVendorId );
    ProductId pId = VendorId( info.hid.dwProductId );
    if ( !device_ )
    {
        device_ = std::make_unique<Device>();
        numMsg_ = 0;

        unsigned pathSize = 0;
        GetRawInputDeviceInfo( handle, RIDI_DEVICENAME, NULL, &pathSize ); // get size
        std::wstring path;
        path.resize( pathSize );

        std::wstring productString;
        productString.resize( 256 );
        std::wstring manufacturerString;
        manufacturerString.resize( 256 );

        bool gotProdStr = false;
        bool gotManufacturerStr = false;

        GetRawInputDeviceInfo( handle, RIDI_DEVICENAME, path.data(), &pathSize ); // get name

        auto hidHandler = CreateFile( path.data(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL );
        if ( hidHandler != INVALID_HANDLE_VALUE )
        {
            gotProdStr = HidD_GetProductString( hidHandler, productString.data(), unsigned( productString.size() ) );
            gotManufacturerStr = HidD_GetManufacturerString( hidHandler, manufacturerString.data(), unsigned( manufacturerString.size() ) );
            CloseHandle( hidHandler );
        }

        if ( gotProdStr && gotManufacturerStr )
        {
            spdlog::info( "WIN API device found: {:04x}:{:04x}, name={}:{}",
                vId, pId, wideToUtf8( manufacturerString.c_str() ), wideToUtf8( productString.c_str() ) );
            TelemetrySignal( fmt::format( "WIN API device {:04x}:{:04x} found: {}:{}",
                vId, pId, wideToUtf8( manufacturerString.c_str() ), wideToUtf8( productString.c_str() ) ) );
        }


        spdlog::info( "SpaceMouse connected: {:04x}:{:04x}, path={}", vId, pId, wideToUtf8( path.c_str() ) );
        TelemetrySignal( fmt::format( "WIN API device {:04x}:{:04x} opened", vId, pId ) );
    }
    device_->updateDevice( vId, pId );
}

}

#endif // _WIN32
