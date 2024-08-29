#include "MRApple.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer.h"
#include "MRViewerEventQueue.h"

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
    if ( err == noErr )
        err = AECountItems( &docList, &itemCount );
    else
        itemCount = 0;

    // Build paths list
    std::vector<std::filesystem::path> paths;
    std::string joined;
    for ( long i = 1; i <= itemCount; ++i )
    {
        err = AEGetNthPtr( &docList, i, typeFSRef, nullptr, nullptr, &fileRef, sizeof( fileRef ), nullptr );
        if ( err != noErr ) continue;
        // Convert the FSRef to a path
        err = FSRefMakePath( &fileRef, ( UInt8* )path, sizeof( path ) );
        if ( err == noErr )
            paths.push_back( path );
        else
            path[0] = '?', path[1] = '\x0';
        if ( !joined.empty() )
            joined += ',';
        joined += path;
    }
    AEDisposeDesc( &docList );

    // Signal to open files
    spdlog::info( "Request to open files: {}", joined );
    MR::getViewerInstance().openFiles( paths );

    if ( err != noErr )
        spdlog::warn( "kAEOpenDocuments error: (OSErr){}", err );
    return err;
}

#pragma clang diagnostic pop

void registerOpenDocumentsCallback()
{
    OSErr err = AEInstallEventHandler( kCoreEventClass, kAEOpenDocuments, NewAEEventHandlerUPP( handleOpenDocuments ), 0, false );
    if ( err != noErr )
        spdlog::warn( "AEInstallEventHandler(...kAEOpenDocuments...) failed: (OSErr){}", err );
}

}

}
#endif
