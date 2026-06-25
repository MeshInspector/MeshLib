#pragma once
#ifdef _WIN32

#include "MRSignalCombiners.h"
#include "MRMesh/MRSignal.h"

namespace MR
{

class Win32MessageHandler : public std::enable_shared_from_this<Win32MessageHandler>
{
    struct Private {};

public:
    Win32MessageHandler( Private, HWND window );
    ~Win32MessageHandler();
    static std::shared_ptr<Win32MessageHandler> getHandler( HWND window );

    bool isValid() const;

    boost::signals2::signal<bool ( HWND window, UINT message, WPARAM wParam, LPARAM lParam ), StopOnTrueCombiner> onMessage;

    static LRESULT WINAPI WindowSubclassProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam );

private:
    HWND window_;
    LONG_PTR parentProc_;
};

} // namespace MR

#endif
