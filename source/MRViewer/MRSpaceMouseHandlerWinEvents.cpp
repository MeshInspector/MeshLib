#include "MRSpaceMouseHandlerWinEvents.h"
#ifdef _WIN32
#include "MRViewer.h"
#include "MRMesh/MRTelemetry.h"
#include "MRWin32MessageHandler.h"

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include "MRPch/MRWinapi.h"
#include "hidusage.h"

namespace MR::SpaceMouse
{

bool HandlerWinEvents::initialize()
{
    auto window = getViewerInstance().window;
    if ( !window )
    {
        assert( false && "no glfw window while trying to initialize SpaceMouse::HandlerWinEvents" );
        return false;
    }
    auto winHndl = glfwGetWin32Window( window );
    if ( !winHndl )
    {
        assert( false && "no window handler while trying to initialize SpaceMouse::HandlerWinEvents" );
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
        assert( false && "invalid Win32MessageHandler while trying to initialize SpaceMouse::HandlerWinEvents" );
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

        RID_DEVICE_INFO info;
        unsigned infoSize = sizeof( RID_DEVICE_INFO );
        info.cbSize = infoSize;

        GetRawInputDeviceInfo( rawInput.header.hDevice, RIDI_DEVICEINFO, &info, &infoSize );

        VendorId vId = VendorId( info.hid.dwVendorId );
        ProductId pId = VendorId( info.hid.dwProductId );
        if ( !device_ )
        {
            device_ = std::make_unique<Device>();
            numMsg_ = 0;
            spdlog::info( "SpaceMouse connected: {:04x}:{:04x}", vId, pId );
            TelemetrySignal( fmt::format( "WIN API device {:04x}:{:04x} opened", vId, pId ) );
        }
        device_->updateDevice( vId, pId );
        
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

}

#endif // _WIN32
