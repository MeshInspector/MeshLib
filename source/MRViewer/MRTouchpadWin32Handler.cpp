#ifdef _WIN32
#include "MRTouchpadWin32Handler.h"

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <spdlog/spdlog.h>

#pragma warning( push )
#pragma warning( disable: 5204 )
#include <comdef.h>
#pragma warning( pop )

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

class HRESULTHandler
{
public:
    explicit HRESULTHandler( unsigned line )
        : line_( line )
    {
        //
    }

    HRESULTHandler& operator =( HRESULT hr )
    {
        if ( hr != S_OK )
        {
            _com_error err( hr );
            _bstr_t msg( err.ErrorMessage() );
            spdlog::error( "Line {}: error code = {:x} message = \"{}\"", line_, hr, (const char*)msg );
        }
        return *this;
    }

private:
    unsigned line_;
};

#define HR HRESULTHandler( __LINE__ )

#define UNUSED( x ) (void)( x )

}

namespace MR
{

class TouchpadWin32Handler::DirectManipulationViewportEventHandler
    : public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::RuntimeClassType::ClassicCom>,
        Microsoft::WRL::Implements<
            Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::RuntimeClassType::ClassicCom>,
            Microsoft::WRL::FtmBase,
            IDirectManipulationViewportEventHandler
        >
    >
{
public:
    explicit DirectManipulationViewportEventHandler( TouchpadWin32Handler* handler )
        : handler_( handler )
    {
        //
    }

    ~DirectManipulationViewportEventHandler() override
    {
        //
    }

    HRESULT STDMETHODCALLTYPE OnViewportStatusChanged(
        IDirectManipulationViewport* viewport,
        DIRECTMANIPULATION_STATUS current,
        DIRECTMANIPULATION_STATUS previous
    ) override
    {
        if ( current == previous )
            return S_OK;
        HR = viewport->ZoomToRect( 0.f, 0.f, 1000.f, 1000.f, FALSE );
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnViewportUpdated(
        IDirectManipulationViewport* viewport
    ) override
    {
        UNUSED( viewport );
        spdlog::info( "OnViewportUpdated" );
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnContentUpdated(
        IDirectManipulationViewport* viewport,
        IDirectManipulationContent* content
    ) override
    {
        UNUSED( viewport );

        float transform[6];
        HR = content->GetContentTransform( transform, ARRAYSIZE( transform ) );
        spdlog::info( "scale = {} x = {} y = {}", transform[0], transform[4], transform[5] );

        return S_OK;
    }

private:
    TouchpadWin32Handler* handler_;
};

TouchpadWin32Handler::TouchpadWin32Handler( GLFWwindow* window )
{
    window_ = glfwGetWin32Window( window );

    TouchpadWin32HandlerRegistry::instance().add( window_, this );

#pragma warning( push )
#pragma warning( disable: 4302 )
#pragma warning( disable: 4311 )
    glfwProc_ = SetWindowLongPtr( window_, GWLP_WNDPROC, ( LONG_PTR )&TouchpadWin32Handler::WindowSubclassProc );
#pragma warning( pop )
    if ( glfwProc_ == 0 )
    {
        spdlog::error( "Failed to set the window procedure: {:x}", GetLastError() );
        return;
    }

    HR = ::CoInitializeEx( NULL, COINIT_APARTMENTTHREADED );
    HR = ::CoCreateInstance( CLSID_DirectManipulationManager, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS( &manager_ ) );
    HR = manager_->GetUpdateManager( IID_PPV_ARGS( &updateManager_ ) );

    HR = manager_->CreateViewport( NULL, window_, IID_PPV_ARGS( &viewport_ ) );
    DIRECTMANIPULATION_CONFIGURATION configuration =
        DIRECTMANIPULATION_CONFIGURATION_INTERACTION |
        DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X |
        DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y |
        DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_INERTIA |
        DIRECTMANIPULATION_CONFIGURATION_RAILS_X |
        DIRECTMANIPULATION_CONFIGURATION_RAILS_Y |
        DIRECTMANIPULATION_CONFIGURATION_SCALING |
        DIRECTMANIPULATION_CONFIGURATION_SCALING_INERTIA;
    HR = viewport_->ActivateConfiguration( configuration );
    HR = viewport_->SetViewportOptions( DIRECTMANIPULATION_VIEWPORT_OPTIONS_MANUALUPDATE );

    eventHandler_ = Microsoft::WRL::Make<DirectManipulationViewportEventHandler>( this );
    HR = viewport_->AddEventHandler( window_, eventHandler_.Get(), &eventHandlerCookie_ );

    const RECT viewportRect {
        .left = 0,
        .top = 0,
        .right = 1000,
        .bottom = 1000,
    };
    HR = viewport_->SetViewportRect( &viewportRect );

    HR = manager_->Activate( window_ );
    HR = viewport_->Enable();
    HR = updateManager_->Update( NULL );
}

TouchpadWin32Handler::~TouchpadWin32Handler()
{
    HR = viewport_->Stop();
    HR = viewport_->Disable();
    HR = viewport_->RemoveEventHandler( eventHandlerCookie_ );
    HR = viewport_->Abandon();

    HR = manager_->Deactivate( window_ );

    SetWindowLongPtr( window_, GWLP_WNDPROC, glfwProc_ );

    TouchpadWin32HandlerRegistry::instance().remove( window_ );
}

LRESULT WINAPI TouchpadWin32Handler::WindowSubclassProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    auto* handler = TouchpadWin32HandlerRegistry::instance().find( hwnd );
    assert( handler );

    switch ( uMsg )
    {
    //case WM_POINTERDOWN:
    case DM_POINTERHITTEST:
        HR = handler->updateManager_->Update( NULL );
        break;
    case WM_INPUT:
        break;
    }

#pragma warning( push )
#pragma warning( disable: 4312 )
    return CallWindowProc( (WNDPROC)handler->glfwProc_, hwnd, uMsg, wParam, lParam );
#pragma warning( pop )
}

}

#endif
