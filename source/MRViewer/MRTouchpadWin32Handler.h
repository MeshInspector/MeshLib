#pragma once
#ifdef _WIN32

#include "MRTouchpadController.h"
#include "MRWin32MessageHandler.h"

#pragma warning( push )
#pragma warning( disable: 4265 )
#pragma warning( disable: 4986 )
#pragma warning( disable: 5204 )
#pragma warning( disable: 5220 )
#include <directmanipulation.h>
#include <wrl.h>
#pragma warning( pop )

#include <map>

namespace MR
{

// Touchpad event handler for Windows using Direct Manipulation API
// More info: https://learn.microsoft.com/en-us/windows/win32/directmanipulation/direct-manipulation-portal
class TouchpadWin32Handler : public TouchpadController::Handler
{
public:
    TouchpadWin32Handler( GLFWwindow* window );
    ~TouchpadWin32Handler() override;

    static void CALLBACK TouchpadEventPoll( PVOID lpParam, BOOLEAN timerOrWaitFired );

private:
    HWND window_;

    std::shared_ptr<Win32MessageHandler> msgHandler_;
    boost::signals2::scoped_connection onWinMsg_;
    void processPointerHitTestEvent_( WPARAM wParam );

    Microsoft::WRL::ComPtr<IDirectManipulationManager> manager_;
    Microsoft::WRL::ComPtr<IDirectManipulationUpdateManager> updateManager_;
    Microsoft::WRL::ComPtr<IDirectManipulationViewport> viewport_;

    class DirectManipulationViewportEventHandler;
    Microsoft::WRL::ComPtr<DirectManipulationViewportEventHandler> eventHandler_;
    DWORD eventHandlerCookie_;

    friend class DirectManipulationViewportEventHandler;
    HANDLE timerQueue_{ NULL };
    HANDLE timer_{ NULL };
    void startTouchpadEventPolling_();
    void stopTouchpadEventPolling_();
};

} // namespace MR

#endif
