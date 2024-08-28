#include "MRApple.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer.h"
#include "MRViewerEventQueue.h"
#include <spdlog/fmt/ranges.h> // TODO: remove when it is in PCH

#ifdef __APPLE__

#include <Carbon/Carbon.h>

namespace MR
{

namespace Apple
{

// Some Carbon functions are deprecated
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

// "documents open" event handler
static OSErr handleOpenDocuments( const AppleEvent* event, AppleEvent* /* reply */, void* /* refCon */ )
{
    AEDescList docList;
    FSRef fileRef;
    long itemCount;
    OSErr err;
    char path[PATH_MAX];

    // Get the list of files to open and obtain its size
    err = AEGetParamDesc( event, keyDirectObject, typeAEList, &docList );
    if ( err != noErr ) return err;
    err = AECountItems( &docList, &itemCount );
    if ( err != noErr ) return err;

    // Build paths list
    std::vector<std::filesystem::path> paths;
    for ( long i = 1; i <= itemCount; ++i )
    {
        err = AEGetNthPtr( &docList, i, typeFSRef, nullptr, nullptr, &fileRef, sizeof( fileRef ), nullptr );
        if ( err != noErr ) continue;
        // Convert the FSRef to a path
        err = FSRefMakePath( &fileRef, ( UInt8* )path, sizeof( path ) );
        if ( err == noErr )
            paths.push_back( path );
    }
    AEDisposeDesc( &docList );

    // Signal to open files
    spdlog::info( fmt::format( "Request to open files: ", fmt::join( paths, "," ) ) );
    MR::getViewerInstance().openFiles( paths );

    return noErr;
}

#pragma clang diagnostic pop

void registerOpenDocumentsCallback()
{
    AEInstallEventHandler( kCoreEventClass, kAEOpenDocuments, NewAEEventHandlerUPP( handleOpenDocuments ), 0, false );
}

}

}
#endif
