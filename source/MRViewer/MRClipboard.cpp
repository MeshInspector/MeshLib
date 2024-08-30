#include "MRClipboard.h"

#ifndef __EMSCRIPTEN__
    #ifdef _WIN32
        #include <MRMesh/MRFinally.h>
    #else
        #include <clip/clip.h>
    #endif
#endif

namespace MR
{

Expected<std::string> GetClipboardText()
{
#if defined( __EMSCRIPTEN__ )
    return "";
#elif defined( _WIN32 )
    // Try opening the clipboard
    if ( !OpenClipboard( nullptr ) )
        return unexpected( "Could not open clipboard" );
    MR_FINALLY { CloseClipboard(); };

    // Get handle of clipboard object for ANSI text
    HANDLE hData = GetClipboardData( CF_TEXT );
    if ( !hData )
        return ""; // no text data

    // Lock the handle to get the actual text pointer
    char* pszText = static_cast< char* >( GlobalLock( hData ) );
    if ( !pszText )
        return "";
    MR_FINALLY { GlobalUnlock( hData ); };

    // Save text in a string class instance
    std::string text( pszText );

    return text;
#else
    std::string text;
    if ( !clip::get_text( text ) )
        return unexpected( "Could not open clipboard" );
    return text;
#endif
}

Expected<void> SetClipboardText( const std::string& text )
{
#if defined( __EMSCRIPTEN__ )
    ( void )text;
#elif defined( _WIN32 )
    HGLOBAL hMem = GlobalAlloc( GMEM_MOVEABLE, text.size() + 1 );
    memcpy( GlobalLock( hMem ), text.c_str(), text.size() + 1 );
    GlobalUnlock( hMem );

    if ( !OpenClipboard( nullptr ) )
        return unexpected( "Could not open clipboard" );
    MR_FINALLY{ CloseClipboard(); };

    if ( !EmptyClipboard() || !SetClipboardData( CF_TEXT, hMem ) )
        return unexpected( "Could not set clipboard" );
#else
    if ( !clip::set_text( text ) )
        return unexpected( "Could not set clipboard" );
#endif
    return {};
}

} // namespace MR
