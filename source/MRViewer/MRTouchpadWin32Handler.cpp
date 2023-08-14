#ifdef _WIN32
#include "MRTouchpadWin32Handler.h"

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <spdlog/spdlog.h>

#pragma warning( push )
#pragma warning( disable: 4191 )
#pragma warning( disable: 4265 )
#pragma warning( disable: 5204 )
#pragma warning( disable: 5220 )
#include <comdef.h>
#include <hidusage.h>
#include <hidpi.h>
#pragma warning( pop )
#pragma comment( lib, "hid.lib" )

namespace
{

class TouchpadWin32HandlerRegistry
{
public:
    static TouchpadWin32HandlerRegistry& instance()
    {
        static TouchpadWin32HandlerRegistry instance;
        return instance;
    }

    void add( HWND view, MR::TouchpadWin32Handler* handler )
    {
        registry_.emplace( view, handler );
    }

    void remove( HWND view )
    {
        registry_.erase( view );
    }

    [[nodiscard]] MR::TouchpadWin32Handler* find( HWND view ) const
    {
        const auto it = registry_.find( view );
        if ( it != registry_.end() )
            return it->second;
        else
            return nullptr;
    }

private:
    std::map<HWND, MR::TouchpadWin32Handler*> registry_;
};

// define missing constants
// see also: https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/windows-precision-touchpad-required-hid-top-level-collections#windows-precision-touchpad-input-reports
constexpr unsigned HID_USAGE_DIGITIZER_CONTACT_ID = 0x51;
constexpr unsigned HID_USAGE_DIGITIZER_CONTACT_COUNT = 0x54;

std::string formatLastError()
{
	auto hr = GetLastError();
	_com_error err( hr );
	_bstr_t str( err.ErrorMessage() );
	return fmt::format( "{} (code {:x}", (const char*)str, hr );
}

}

namespace MR
{

TouchpadWin32Handler::TouchpadWin32Handler( GLFWwindow* window )
{
    window_ = glfwGetWin32Window( window );

    TouchpadWin32HandlerRegistry::instance().add( window_, this );

#pragma warning( push )
#pragma warning( disable: 4302 )
#pragma warning( disable: 4311 )
    glfwProc_ = SetWindowLongPtr( window_, GWLP_WNDPROC, (LONG_PTR)&TouchpadWin32Handler::WindowSubclassProc );
#pragma warning( pop )
    if ( glfwProc_ == 0 )
    {
        spdlog::error( "Failed to set the window procedure: {}", formatLastError() );
        return;
    }

    RAWINPUTDEVICE rid {
        .usUsagePage = HID_USAGE_PAGE_DIGITIZER,
        .usUsage = HID_USAGE_DIGITIZER_TOUCH_PAD,
        .dwFlags = RIDEV_INPUTSINK,
        .hwndTarget = window_,
    };
    auto rc = RegisterRawInputDevices( &rid, 1, sizeof( RAWINPUTDEVICE ) );
    if ( rc == FALSE )
    {
        spdlog::error( "Failed to register input devices: {}", formatLastError() );
        return;
    }

    fetchDeviceInfo_();
}

TouchpadWin32Handler::~TouchpadWin32Handler()
{
    SetWindowLongPtr( window_, GWLP_WNDPROC, glfwProc_ );

    TouchpadWin32HandlerRegistry::instance().remove( window_ );
}

LRESULT WINAPI TouchpadWin32Handler::WindowSubclassProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    auto* handler = TouchpadWin32HandlerRegistry::instance().find( hwnd );
    assert( handler );

	switch ( uMsg )
	{
    case WM_INPUT:
        processRawInput( *handler, (HRAWINPUT)lParam );
        break;
	}

#pragma warning( push )
#pragma warning( disable: 4312 )
    return CallWindowProc( (WNDPROC)handler->glfwProc_, hwnd, uMsg, wParam, lParam );
#pragma warning( pop )
}

#define CHECK_LAST_ERROR( CALL ) \
if ( ( CALL ) == (UINT)-1 ) \
{ \
    spdlog::error( "Line {}: Failed to fetch device info: {}", __LINE__, formatLastError() ); \
    return; \
}

#define CHECK_NTSTATUS( CALL ) \
if ( NTSTATUS rc = ( CALL ); rc != HIDP_STATUS_SUCCESS ) \
{ \
    spdlog::error( "Line {}: Failed to fetch HID info: {:x}", __LINE__, rc ); \
    return; \
}

void TouchpadWin32Handler::fetchDeviceInfo_()
{
    UINT deviceListSize = 0;
    CHECK_LAST_ERROR( GetRawInputDeviceList( NULL, &deviceListSize, sizeof( RAWINPUTDEVICELIST ) ) )
    std::vector<RAWINPUTDEVICELIST> deviceList( deviceListSize );
    CHECK_LAST_ERROR( GetRawInputDeviceList( deviceList.data(), &deviceListSize, sizeof( RAWINPUTDEVICELIST ) ) )
    for ( const auto& device : deviceList )
    {
        if ( device.dwType != RIM_TYPEHID )
            continue;

        DeviceInfo info;

        UINT deviceNameSize = 0;
        CHECK_LAST_ERROR( GetRawInputDeviceInfo( device.hDevice, RIDI_DEVICENAME, NULL, &deviceNameSize ) )
        std::wstring wDeviceName( deviceNameSize + 1, L'\0' );
        CHECK_LAST_ERROR( GetRawInputDeviceInfo( device.hDevice, RIDI_DEVICENAME, wDeviceName.data(), &deviceNameSize ) )
        _bstr_t bDeviceName( wDeviceName.data() );
        info.deviceName = std::string( ( const char* )bDeviceName );
        spdlog::info( "Found input device: {}", info.deviceName );

        UINT preparsedSize = 0;
        CHECK_LAST_ERROR( GetRawInputDeviceInfo( device.hDevice, RIDI_PREPARSEDDATA, NULL, &preparsedSize ) )
        info.preparsedData.resize( preparsedSize, '\0' );
        CHECK_LAST_ERROR( GetRawInputDeviceInfo( device.hDevice, RIDI_PREPARSEDDATA, (LPVOID)info.preparsedData.data(), &preparsedSize ) )
        auto preparsed = ( PHIDP_PREPARSED_DATA )info.preparsedData.data();

        HIDP_CAPS caps;
        CHECK_NTSTATUS( HidP_GetCaps( preparsed, &caps ) )

        if ( caps.NumberInputValueCaps != 0 )
        {
            USHORT valueCapsSize = caps.NumberInputValueCaps;
            std::vector<HIDP_VALUE_CAPS> valueCaps( valueCapsSize );
            CHECK_NTSTATUS( HidP_GetValueCaps( HidP_Input, valueCaps.data(), &valueCapsSize, preparsed ) )

            for ( const auto& cap : valueCaps )
            {
                if ( cap.IsRange || !cap.IsAbsolute )
                    continue;

                auto& result = info.caps[cap.LinkCollection];
                switch ( cap.UsagePage )
                {
                case HID_USAGE_PAGE_GENERIC:
                    switch ( cap.NotRange.Usage )
                    {
                    case HID_USAGE_GENERIC_X:
                        result.hasX = true;
                        break;
                    case HID_USAGE_GENERIC_Y:
                        result.hasY = true;
                        break;
                    }
                    break;
                case HID_USAGE_PAGE_DIGITIZER:
                    switch ( cap.NotRange.Usage )
                    {
                    case HID_USAGE_DIGITIZER_CONTACT_ID:
                        result.hasContactId = true;
                        break;
                    case HID_USAGE_DIGITIZER_CONTACT_COUNT:
                        info.contactCountLinkCollection = cap.LinkCollection;
                        break;
                    }
                    break;
                }
            }
        }

        if ( caps.NumberInputButtonCaps != 0 )
        {
            USHORT buttonCapsSize = caps.NumberInputButtonCaps;
            std::vector<HIDP_BUTTON_CAPS> buttonCaps( buttonCapsSize );
            CHECK_NTSTATUS( HidP_GetButtonCaps( HidP_Input, buttonCaps.data(), &buttonCapsSize, preparsed ) )

            for ( const auto& cap : buttonCaps )
            {
                if ( cap.IsRange )
                    continue;

                auto& result = info.caps[cap.LinkCollection];
                switch ( cap.UsagePage )
                {
                case HID_USAGE_PAGE_DIGITIZER:
                    switch ( cap.NotRange.Usage )
                    {
                    case HID_USAGE_DIGITIZER_TIP_SWITCH:
                        result.hasTipSwitch = true;
                        break;
                    }
                    break;
                }
            }
        }

        devices_.emplace( device.hDevice, info );
    }
}

void TouchpadWin32Handler::processRawInput( TouchpadWin32Handler& handler, HRAWINPUT hRawInput )
{
	UINT rawInputSize = 0;
	CHECK_LAST_ERROR( GetRawInputData( hRawInput, RID_INPUT, NULL, &rawInputSize, sizeof( RAWINPUTHEADER ) ) )
    auto rawInputData = std::make_unique<unsigned char[]>( rawInputSize );
	CHECK_LAST_ERROR( GetRawInputData( hRawInput, RID_INPUT, (LPVOID)rawInputData.get(), &rawInputSize, sizeof( RAWINPUTHEADER ) ) )

    auto rawInput = (PRAWINPUT)rawInputData.get();
    if ( rawInput->header.dwType != RIM_TYPEHID )
        return;

    auto it = handler.devices_.find( rawInput->header.hDevice );
    if ( it == handler.devices_.end() )
        return;
    const auto& info = it->second;

    auto& hidData = rawInput->data.hid;
    if ( hidData.dwCount == 0 )
        return;

    auto preparsed = (PHIDP_PREPARSED_DATA)info.preparsedData.data();

    for ( const auto& [linkCollection, cap] : info.caps )
    {
        if ( !( cap.hasContactId && cap.hasX && cap.hasY && cap.hasTipSwitch ) )
            continue;

        ULONG touchId = 0, x = 0, y = 0;
        CHECK_NTSTATUS( HidP_GetUsageValue( HidP_Input, HID_USAGE_PAGE_DIGITIZER, linkCollection, HID_USAGE_DIGITIZER_CONTACT_ID, &touchId, preparsed, (PCHAR)hidData.bRawData, hidData.dwSizeHid ) )
        CHECK_NTSTATUS( HidP_GetUsageValue( HidP_Input, HID_USAGE_PAGE_GENERIC, linkCollection, HID_USAGE_GENERIC_X, &x, preparsed, (PCHAR)hidData.bRawData, hidData.dwSizeHid ) )
        CHECK_NTSTATUS( HidP_GetUsageValue( HidP_Input, HID_USAGE_PAGE_GENERIC, linkCollection, HID_USAGE_GENERIC_Y, &y, preparsed, (PCHAR)hidData.bRawData, hidData.dwSizeHid ) )

        auto maxUsageCount = HidP_MaxUsageListLength( HidP_Input, HID_USAGE_PAGE_DIGITIZER, preparsed );
        std::vector<USAGE> usages( maxUsageCount );
        CHECK_NTSTATUS( HidP_GetUsages( HidP_Input, HID_USAGE_PAGE_DIGITIZER, linkCollection, usages.data(), &maxUsageCount, preparsed, ( PCHAR )hidData.bRawData, hidData.dwSizeHid ) )
        bool pressed = std::find( usages.begin(), usages.end(), HID_USAGE_DIGITIZER_TIP_SWITCH ) != usages.end();

        const MR::Vector2ll pos( x, y );
        auto state = handler.state_.find( touchId );
        if ( state == handler.state_.end() )
        {
            if ( pressed )
            {
                spdlog::info( "touch begin: id = {} x = {} y = {}", touchId, x, y );
                handler.state_.emplace( touchId, pos );
            }
            else
            {
                // phantom touch?
            }
        }
        else
        {
            if ( pressed )
            {
                if ( pos != state->second )
                {
					spdlog::info( "touch moved: id = {} x = {} y = {}", touchId, x, y );
                    state->second = pos;
                }
            }
            else
            {
                spdlog::info( "touch end: id = {} x = {} y = {}", touchId, x, y );
                handler.state_.erase( touchId );
            }
        }
    }
}

}

#endif
