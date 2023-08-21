#ifdef _WIN32
#include "MRTouchpadWin32Handler.h"

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <spdlog/spdlog.h>

#define EVENT_DEBUG

#ifdef EVENT_DEBUG
#pragma warning( push )
#pragma warning( disable: 5204 )
#include <comdef.h>
#pragma warning( pop )
#endif

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
    explicit HRESULTHandler( std::string_view file, unsigned line )
        : file_( file )
        , line_( line )
        , hr_( S_OK )
    {
        //
    }

    HRESULTHandler& operator =( [[maybe_unused]] HRESULT hr )
    {
        hr_ = hr;
#ifdef EVENT_DEBUG
        if ( hr != S_OK )
        {
            _com_error err( hr );
            _bstr_t msg( err.ErrorMessage() );
            if ( SUCCEEDED( hr ) )
                spdlog::warn( "{}:{}: {:08x} {}", file_, line_, (unsigned long)hr, (const char*)msg );
            else
                spdlog::error( "{}:{}: {:08x} {}", file_, line_, (unsigned long)hr, (const char*)msg );
        }
#endif
        return *this;
    }

    operator HRESULT() const
    {
        return hr_;
    }

private:
    std::string_view file_;
    unsigned line_;
    HRESULT hr_;
};

consteval std::string_view FILENAME( std::string_view str )
{
    if ( auto posU = str.rfind( '/' ); posU != std::string_view::npos )
        return str.substr( posU + 1 );
    else if ( auto posW = str.rfind( '\\' ); posW != std::string_view::npos )
        return str.substr( posW + 1 );
    else
        return str;
}

#define HR HRESULTHandler( FILENAME( __FILE__ ), __LINE__ )

#define UNUSED( x ) (void)( x )

constexpr DWORD TOUCHPAD_EVENT_POLLING_PERIOD_MS = 10; // 100 Hz

#define FUZZY( x, y ) ( std::abs( ( x ) - ( y ) ) < 1e-6 )
#define FUZZY_0( x ) FUZZY( ( x ), 0 )
#define FUZZY_1( x ) FUZZY( ( x ), 1 )

}

namespace MR
{

class TouchpadWin32Handler::DirectManipulationViewportEventHandler
    : public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::RuntimeClassType::ClassicCom>,
        Microsoft::WRL::Implements<
            Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::RuntimeClassType::ClassicCom>,
            Microsoft::WRL::FtmBase,
            IDirectManipulationViewportEventHandler,
            IDirectManipulationInteractionEventHandler
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

    static std::string toString( DIRECTMANIPULATION_STATUS status )
    {
        switch ( status )
        {
#define CASE(STATUS) case DIRECTMANIPULATION_##STATUS: return #STATUS;
        CASE(BUILDING)
        CASE(ENABLED)
        CASE(DISABLED)
        CASE(RUNNING)
        CASE(INERTIA)
        CASE(READY)
        CASE(SUSPENDED)
#undef CASE
        }
        assert( false );
        return {};
    }

    HRESULT STDMETHODCALLTYPE OnViewportStatusChanged(
        IDirectManipulationViewport* viewport,
        DIRECTMANIPULATION_STATUS current,
        DIRECTMANIPULATION_STATUS previous
    ) override
    {
        assert( status_ == previous );
        if ( current == previous )
            return S_OK;
        status_ = current;

#ifdef EVENT_DEBUG
        spdlog::info( "touchpad gesture state changed: {} -> {}", toString( previous ), toString( current ) );
#endif
        if ( current == DIRECTMANIPULATION_READY )
        {
            endGesture_();
            HR = viewport->ZoomToRect( 0.f, 0.f, 1000.f, 1000.f, FALSE );
        }

        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnViewportUpdated(
        IDirectManipulationViewport* viewport
    ) override
    {
        UNUSED( viewport );
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

        const auto
            scaleX  = transform[0],
            rotateY = transform[1],
            rotateX = transform[2],
            scaleY  = transform[3],
            offsetX = transform[4],
            offsetY = transform[5];
        if ( !FUZZY_1( scaleX ) || !FUZZY_1( scaleY ) )
        {
            assert( scaleX == scaleY );
            const auto scale = scaleX;
            if ( !FUZZY( scale, lastScale_ ) )
            {
                emitZoom_( scale );
                lastScale_ = scale;
            }
        }
#ifdef EVENT_DEBUG
        else if ( !FUZZY_0( rotateX ) | !FUZZY_0( rotateY ) )
        {
            spdlog::info( "rotate x = {} y = {}", rotateX, rotateY );
        }
#endif
        else if ( !FUZZY_0( offsetX ) || !FUZZY_0( offsetY ) )
        {
            if ( !FUZZY( offsetX, lastOffsetX_ ) || !FUZZY( offsetY, lastOffsetY_ ) )
            {
                emitSwipe_( offsetX - lastOffsetX_, offsetY - lastOffsetY_ );
                lastOffsetX_ = offsetX;
                lastOffsetY_ = offsetY;
            }
        }

        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnInteraction(
        IDirectManipulationViewport2* viewport,
        DIRECTMANIPULATION_INTERACTION_TYPE interaction
    ) override
    {
        UNUSED( viewport );

        if ( interaction == DIRECTMANIPULATION_INTERACTION_BEGIN )
        {
#ifdef EVENT_DEBUG
            spdlog::info( "touchpad gesture interaction started" );
#endif
            handler_->startTouchpadEventPolling_();
        }
        else if ( interaction == DIRECTMANIPULATION_INTERACTION_END )
        {
#ifdef EVENT_DEBUG
            spdlog::info( "touchpad gesture interaction finished" );
#endif
            handler_->stopTouchpadEventPolling_();
        }

        return S_OK;
    }

    inline DIRECTMANIPULATION_STATUS getStatus() const
    {
        return status_;
    }

private:
    TouchpadWin32Handler* handler_;
    DIRECTMANIPULATION_STATUS status_{ DIRECTMANIPULATION_BUILDING };

    enum class Gesture
    {
        None,
        Rotate,
        Swipe,
        Zoom,
    } gesture_{ Gesture::None };
    float lastAngle_{ 0.f };
    float lastOffsetX_{ 0.f };
    float lastOffsetY_{ 0.f };
    float lastScale_{ 1.f };

    void beginGesture_()
    {
        switch ( gesture_ )
        {
        case Gesture::None:
            break;
        case Gesture::Rotate:
            handler_->rotate( 0.f, TouchpadController::Handler::GestureState::Begin );
            break;
        case Gesture::Swipe:
            break;
        case Gesture::Zoom:
            handler_->zoom( 1.f, TouchpadController::Handler::GestureState::Begin );
            break;
        }
    }

    void endGesture_()
    {
        switch ( gesture_ )
        {
        case Gesture::None:
            break;
        case Gesture::Rotate:
            handler_->rotate( lastAngle_, TouchpadController::Handler::GestureState::End );
            break;
        case Gesture::Swipe:
            break;
        case Gesture::Zoom:
            handler_->zoom( lastScale_, TouchpadController::Handler::GestureState::End );
            break;
        }

        gesture_ = Gesture::None;
        lastAngle_ = 0.f;
        lastOffsetX_ = 0.f;
        lastOffsetY_ = 0.f;
        lastScale_ = 1.f;
    }

    void updateGesture_( Gesture gesture )
    {
        if ( gesture_ != gesture )
        {
            endGesture_();
            gesture_ = gesture;
            beginGesture_();
        }
    }

    void emitSwipe_( float dx, float dy )
    {
        updateGesture_( Gesture::Swipe );
        const auto kinetic = status_ == DIRECTMANIPULATION_INERTIA;
        handler_->swipe( dx, dy, kinetic );
    }

    void emitZoom_( float scale )
    {
        updateGesture_( Gesture::Zoom );
        const auto kinetic = status_ == DIRECTMANIPULATION_INERTIA;
        // TODO: support kinetic zoom
        if ( kinetic )
            return;
        handler_->zoom( scale, TouchpadController::Handler::GestureState::Change );
    }
};

TouchpadWin32Handler::TouchpadWin32Handler( GLFWwindow* window )
{
    window_ = glfwGetWin32Window( window );

    TouchpadWin32HandlerRegistry::instance().add( window_, this );

    timerQueue_ = CreateTimerQueue();

#pragma warning( push )
#pragma warning( disable: 4302 )
#pragma warning( disable: 4311 )
    glfwProc_ = SetWindowLongPtr( window_, GWLP_WNDPROC, ( LONG_PTR )&TouchpadWin32Handler::WindowSubclassProc );
#pragma warning( pop )
    if ( glfwProc_ == 0 )
    {
        spdlog::error( "Failed to set the window procedure (code {:08x})", GetLastError() );
        return;
    }

#define CHECK( EXPR ) \
    if ( HRESULT hr = EXPR; !SUCCEEDED( hr ) ) \
    { \
        spdlog::error( "Failed to initialize touchpad event processing (code {:08x})", hr ); \
        return; \
    }

    CHECK( HR = ::CoInitializeEx( NULL, COINIT_APARTMENTTHREADED ) )
    CHECK( HR = ::CoCreateInstance( CLSID_DirectManipulationManager, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS( &manager_ ) ) )
    CHECK( HR = manager_->GetUpdateManager( IID_PPV_ARGS( &updateManager_ ) ) )

    CHECK( HR = manager_->CreateViewport( NULL, window_, IID_PPV_ARGS( &viewport_ ) ) )
    DIRECTMANIPULATION_CONFIGURATION configuration =
        DIRECTMANIPULATION_CONFIGURATION_INTERACTION |
        DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X |
        DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y |
        DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_INERTIA |
        DIRECTMANIPULATION_CONFIGURATION_RAILS_X |
        DIRECTMANIPULATION_CONFIGURATION_RAILS_Y |
        DIRECTMANIPULATION_CONFIGURATION_SCALING |
        DIRECTMANIPULATION_CONFIGURATION_SCALING_INERTIA;
    CHECK( HR = viewport_->ActivateConfiguration( configuration ) )
    CHECK( HR = viewport_->SetViewportOptions( DIRECTMANIPULATION_VIEWPORT_OPTIONS_MANUALUPDATE ) )

    eventHandler_ = Microsoft::WRL::Make<DirectManipulationViewportEventHandler>( this );
    CHECK( HR = viewport_->AddEventHandler( window_, eventHandler_.Get(), &eventHandlerCookie_ ) )

    const RECT viewportRect {
        .left = 0,
        .top = 0,
        .right = 1000,
        .bottom = 1000,
    };
    CHECK( HR = viewport_->SetViewportRect( &viewportRect ) )

    CHECK( HR = manager_->Activate( window_ ) )
    CHECK( HR = viewport_->Enable() )
}

TouchpadWin32Handler::~TouchpadWin32Handler()
{
    HR = viewport_->Stop();
    HR = viewport_->Disable();
    HR = viewport_->RemoveEventHandler( eventHandlerCookie_ );
    HR = viewport_->Abandon();

    HR = manager_->Deactivate( window_ );

    SetWindowLongPtr( window_, GWLP_WNDPROC, glfwProc_ );

    DeleteTimerQueue( timerQueue_ );

    TouchpadWin32HandlerRegistry::instance().remove( window_ );
}

LRESULT WINAPI TouchpadWin32Handler::WindowSubclassProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    auto* handler = TouchpadWin32HandlerRegistry::instance().find( hwnd );
    assert( handler );

    switch ( uMsg )
    {
    case DM_POINTERHITTEST:
        handler->processPointerHitTestEvent_( wParam );
        break;
    case WM_INPUT:
        break;
    }

#pragma warning( push )
#pragma warning( disable: 4312 )
    return CallWindowProc( (WNDPROC)handler->glfwProc_, hwnd, uMsg, wParam, lParam );
#pragma warning( pop )
}

void TouchpadWin32Handler::processPointerHitTestEvent_( WPARAM wParam )
{
    auto pointerId = GET_POINTERID_WPARAM( wParam );
    POINTER_INPUT_TYPE pointerInputType;
    if ( !::GetPointerType( pointerId, &pointerInputType ) )
        return;
    if ( pointerInputType != PT_TOUCHPAD )
        return;

    viewport_->SetContact( pointerId );
}

void TouchpadWin32Handler::TouchpadEventPoll( PVOID lpParam, BOOLEAN timerOrWaitFired )
{
    UNUSED( timerOrWaitFired );
    if ( lpParam == NULL )
        return;

    auto* handler = (TouchpadWin32Handler*)lpParam;
    const auto status = handler->eventHandler_->getStatus();
    if ( status == DIRECTMANIPULATION_RUNNING || status == DIRECTMANIPULATION_INERTIA )
        HR = handler->updateManager_->Update( NULL );
}

void TouchpadWin32Handler::startTouchpadEventPolling_()
{
    if ( timer_ == NULL )
    {
        auto result = CreateTimerQueueTimer( &timer_, timerQueue_, ( WAITORTIMERCALLBACK )&TouchpadWin32Handler::TouchpadEventPoll, this, 0, TOUCHPAD_EVENT_POLLING_PERIOD_MS, WT_EXECUTEINTIMERTHREAD );
        UNUSED( result );
        assert( timer_ != NULL );
    }
}

void TouchpadWin32Handler::stopTouchpadEventPolling_()
{
    if ( timer_ != NULL )
    {
        auto result = DeleteTimerQueueTimer( timerQueue_, timer_, NULL );
        UNUSED( result );
        timer_ = NULL;
    }
}

}

#endif
