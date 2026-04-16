#ifdef _WIN32
#include "MRWin32MessageHandler.h"

#include "MRMesh/MRMeshFwd.h"
#include "MRPch/MRSpdlog.h"

namespace
{

MR::HashMap<HWND, MR::Win32MessageHandler*> gRegistry;

}

namespace MR
{

Win32MessageHandler::~Win32MessageHandler()
{
    if ( parentProc_ != 0 )
        SetWindowLongPtr( window_, GWLP_WNDPROC, parentProc_ );

    assert( gRegistry[window_] == this );
    gRegistry[window_] = nullptr;
}

std::shared_ptr<Win32MessageHandler> Win32MessageHandler::getHandler( HWND window )
{
    if ( auto* handler = gRegistry[window] )
        return handler->shared_from_this();

    auto handler = std::make_shared<Win32MessageHandler>( Private{}, window );
    if ( !handler->isValid() )
        return {};
    return handler;
}

bool Win32MessageHandler::isValid() const
{
    return parentProc_ != 0;
}

LRESULT Win32MessageHandler::WindowSubclassProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    auto* handler = gRegistry[hwnd];
    assert( handler );
    if ( handler->onMessage( hwnd, uMsg, wParam, lParam ) )
        return TRUE;

#pragma warning( push )
#pragma warning( disable: 4312 )
    return CallWindowProc( (WNDPROC)handler->parentProc_, hwnd, uMsg, wParam, lParam );
#pragma warning( pop )
}

Win32MessageHandler::Win32MessageHandler( Private, HWND window )
    : window_( window )
{
    gRegistry[window_] = this;

#pragma warning( push )
#pragma warning( disable: 4302 )
#pragma warning( disable: 4311 )
    parentProc_ = SetWindowLongPtr( window_, GWLP_WNDPROC, ( LONG_PTR )&Win32MessageHandler::WindowSubclassProc );
#pragma warning( pop )
    if ( parentProc_ == 0 )
    {
        spdlog::warn( "Failed to set the window procedure (code {:08x})", GetLastError() );
        return;
    }
}

} // namespace MR

#endif
