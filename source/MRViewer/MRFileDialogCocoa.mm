#include "MRFileDialogCocoa.h"

#include "MRMesh/MRFinally.h"

#include <AppKit/AppKit.h>

namespace
{

using namespace MR;

inline std::string fromNSURL( NSURL* url )
{
    return [[url path] UTF8String];
}

inline NSString* toNSString( const std::string& path )
{
    return [NSString stringWithUTF8String:path.c_str()];
}

inline NSURL* toNSURL( const std::string& path )
{
    return [NSURL fileURLWithPath:toNSString( path )];
}

NSView* createAccessoryView( const IOFilters& filters )
{
    auto* label = [NSTextField labelWithString:@"Format:"];

    auto* popup = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    for ( const auto& filter : filters )
        [popup addItemWithTitle:toNSString( filter.name )];

    auto* view = [[NSView alloc] initWithFrame:NSZeroRect];
    [view addSubview:label];
    [view addSubview:popup];

    // compute widget layout
    [label sizeToFit];
    [popup sizeToFit];
    const auto labelSize = [label frame].size;
    const auto popupSize = [popup frame].size;
    const auto maxHeight = std::max( labelSize.height, popupSize.height );

    constexpr auto cPadding = 8;
    constexpr auto cSpacing = 4;
#define _ cPadding
#define __ cSpacing
    [view setFrameSize:NSMakeSize(
        _ + labelSize.width + __ + popupSize.width + _,
        _ + maxHeight + _
    )];
    [label setFrameOrigin:NSMakePoint(
        _,
        _ + ( maxHeight - labelSize.height ) / 2
    )];
    [popup setFrameOrigin:NSMakePoint(
        _ + labelSize.width + __,
        _ + ( maxHeight - popupSize.height ) / 2
    )];
#undef __
#undef _

    return view;
}

} // namespace

namespace MR::detail
{

std::vector<std::filesystem::path> runCocoaFileDialog( const FileDialogParameters& params )
{
    // enable garbage collector
    auto* pool = [[NSAutoreleasePool alloc] init];
    MR_FINALLY {
        [pool release];
    };

    // set focus back to the main window at the end
    auto* keyWindow = [[NSApplication sharedApplication] keyWindow];
    MR_FINALLY {
        [keyWindow makeKeyAndOrderFront:nil];
    };

    NSSavePanel* dialog;
    if ( params.saveDialog )
    {
        dialog = [NSSavePanel savePanel];
    }
    else
    {
        auto* openDialog = [NSOpenPanel openPanel];

        [openDialog setAllowsMultipleSelection:params.multiselect];
        [openDialog setCanChooseDirectories:params.folderDialog];
        [openDialog setCanChooseFiles:!params.folderDialog];

        dialog = openDialog;
    }

    const auto currentFolder = getCurrentFolder( params.baseFolder );
    if ( !currentFolder.empty() )
        [dialog setDirectoryURL:toNSURL( currentFolder )];

    if ( !params.fileName.empty() )
        [dialog setNameFieldStringValue:toNSString( params.fileName )];

    if ( !params.folderDialog && !params.filters.empty() )
    {
        auto* fileTypes = [[NSMutableArray alloc] init];
        for ( const auto& filter : params.filters )
        {
            size_t separatorPos = 0;
            for (;;)
            {
                auto nextSeparatorPos = filter.extensions.find( ";", separatorPos );
                auto ext = filter.extensions.substr( separatorPos, nextSeparatorPos - separatorPos );

                assert( ext.starts_with( "*." ) );
                if ( ext != "*.*" )
                    [fileTypes addObject:toNSString( ext.substr( 2 ) )];

                if ( nextSeparatorPos == std::string::npos )
                    break;
                separatorPos = nextSeparatorPos + 1;
            }
        }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        [dialog setAllowedFileTypes:fileTypes];
#pragma clang diagnostic pop

        dialog.accessoryView = createAccessoryView( params.filters );
    }

    if ( [dialog runModal] != NSModalResponseOK )
        return {};

    std::vector<std::filesystem::path> results;
    if ( !params.saveDialog && params.multiselect )
    {
        auto* urls = [(NSOpenPanel*)dialog URLs];
        for ( NSURL* url in urls )
            results.emplace_back( fromNSURL( url ) );
    }
    else
    {
        auto* url = [dialog URL];
        results.emplace_back( fromNSURL( url ) );
    }
    return results;
}

} // namespace MR::detail
