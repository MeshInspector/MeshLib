#ifdef _WIN32
#include "MRDragDropWin32Handler.h"
#include "MRViewer.h"
#include "MRPch/MRSpdlog.h"


#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#pragma warning( push )
#pragma warning( disable: 5204 )
#include <oleidl.h>
#include <ole2.h>
#pragma warning( pop )

namespace MR
{

#pragma warning( push )
#pragma warning( disable: 5204 )
class WinDropTarget : public IDropTarget
{
public:
    virtual HRESULT STDMETHODCALLTYPE QueryInterface(
    /* [in] */ REFIID riid,
    /* [iid_is][out] */ _COM_Outptr_ void __RPC_FAR* __RPC_FAR* ppvObject ) override;

    virtual ULONG STDMETHODCALLTYPE AddRef( void ) override;

    virtual ULONG STDMETHODCALLTYPE Release( void ) override;

    virtual HRESULT STDMETHODCALLTYPE DragEnter(
            /* [unique][in] */ __RPC__in_opt IDataObject* pDataObj,
            /* [in] */ DWORD grfKeyState,
            /* [in] */ POINTL pt,
            /* [out][in] */ __RPC__inout DWORD* pdwEffect ) override;

    virtual HRESULT STDMETHODCALLTYPE DragOver(
        /* [in] */ DWORD grfKeyState,
        /* [in] */ POINTL pt,
        /* [out][in] */ __RPC__inout DWORD* pdwEffect ) override;

    virtual HRESULT STDMETHODCALLTYPE DragLeave( void ) override;

    virtual HRESULT STDMETHODCALLTYPE Drop(
        /* [unique][in] */ __RPC__in_opt IDataObject* pDataObj,
        /* [in] */ DWORD grfKeyState,
        /* [in] */ POINTL pt,
        /* [out][in] */ __RPC__inout DWORD* pdwEffect ) override;
private:
    int64_t owners_ = 0;
};
#pragma warning( pop )

HRESULT STDMETHODCALLTYPE WinDropTarget::QueryInterface(/* [in] */ REFIID riid, /* [iid_is][out] */ _COM_Outptr_ void __RPC_FAR* __RPC_FAR* ppvObject )
{
    HRESULT result = E_NOINTERFACE;
    *ppvObject = NULL;

    if ( riid == IID_IUnknown || riid == IID_IDropTarget )
    {
        *ppvObject = ( IUnknown* )this;
        this->AddRef();
        result = S_OK;
    }

    return result;
}

ULONG STDMETHODCALLTYPE WinDropTarget::AddRef( void )
{
    return ULONG( InterlockedIncrement64( &owners_ ) );
}

ULONG STDMETHODCALLTYPE WinDropTarget::Release( void )
{
    auto res = ULONG( InterlockedDecrement64( &owners_ ) );
    if ( owners_ == 0 )
    {
        // clear anything
    }
    return res;
}

HRESULT STDMETHODCALLTYPE WinDropTarget::DragEnter(/* [unique][in] */ __RPC__in_opt IDataObject* pDataObj, /* [in] */ DWORD grfKeyState, /* [in] */ POINTL pt, /* [out][in] */ __RPC__inout DWORD* pdwEffect )
{
    if ( !pdwEffect )
        return E_INVALIDARG;
    *pdwEffect = DROPEFFECT_NONE;
    ( void )pDataObj;
    ( void )grfKeyState;
    ( void )pt;

    auto& v = getViewerInstance();
    v.emplaceEvent( "Drag enter", [&v] ()
    {
        v.dragEntranceSignal( true );
    } );

    return S_OK;
}

HRESULT STDMETHODCALLTYPE WinDropTarget::DragOver(/* [in] */ DWORD grfKeyState, /* [in] */ POINTL pt, /* [out][in] */ __RPC__inout DWORD* pdwEffect )
{
    if ( !pdwEffect )
        return E_INVALIDARG;
    *pdwEffect = DROPEFFECT_NONE;
    ( void )grfKeyState;

    auto& v = getViewerInstance();
    v.emplaceEvent( "Drag over", [&v,x = pt.x,y = pt.y] () mutable
    {
        int posx = 0, posy = 0;
        glfwGetWindowPos( v.window, &posx, &posy );
        x -= posx;
        y -= posy;
        v.dragOverSignal( int( std::round( x * v.pixelRatio ) ), int( std::round( y * v.pixelRatio ) ) );
    }, true );

    return S_OK;
}

HRESULT STDMETHODCALLTYPE WinDropTarget::DragLeave( void )
{
    auto& v = getViewerInstance();
    v.emplaceEvent( "Drag leave", [&v] ()
    {
        v.dragEntranceSignal( false );
    } );

    return S_OK;
}

HRESULT STDMETHODCALLTYPE WinDropTarget::Drop(/* [unique][in] */ __RPC__in_opt IDataObject* pDataObj, /* [in] */ DWORD grfKeyState, /* [in] */ POINTL pt, /* [out][in] */ __RPC__inout DWORD* pdwEffect )
{
    if ( !pdwEffect )
        return E_INVALIDARG;
    *pdwEffect = DROPEFFECT_NONE;
    assert( false );
    // glfw_drop_callback still takes this, so we are never here
    spdlog::warn( "Windows drag&drop handler overtook \"Drop\" event" );
    ( void )pDataObj;
    ( void )grfKeyState;
    ( void )pt;
    return S_OK;
}

DragDropWin32Handler::DragDropWin32Handler( GLFWwindow* window )
{
    auto res = OleInitialize( NULL );
    if ( res != S_OK )
        return;

    winDropTartget_ = std::make_unique<WinDropTarget>();

    assert( window );

    window_ = glfwGetWin32Window( window );

    res = RegisterDragDrop( window_, ( IDropTarget* )winDropTartget_.get() );
    if ( res != S_OK )
    {
        winDropTartget_.reset();
    }
}

DragDropWin32Handler::~DragDropWin32Handler()
{
    if ( !winDropTartget_ )
        return;
    RevokeDragDrop( window_ );
}

}
#endif
