#include "MRFileDialogCocoa.h"

#include "MRMesh/MRFinally.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRStringConvert.h"

#include <AppKit/AppKit.h>

namespace
{

using namespace MR;

inline std::string fromNSURL( NSURL* url )
{
    return [[url path] UTF8String];
}

inline NSString* toNSString( const std::string_view& path )
{
    return [[NSString alloc] initWithBytes:path.data() length:path.size() encoding:NSUTF8StringEncoding];
}

inline NSString* toNSString( const std::string& path )
{
    return [NSString stringWithUTF8String:path.c_str()];
}

inline NSURL* toNSURL( const std::string& path )
{
    return [NSURL fileURLWithPath:toNSString( path )];
}

NSArray<NSString*>* makeFileTypes( const IOFilters& filters )
{
    auto* fileTypes = [[NSMutableArray alloc] init];
    for ( const auto& filter : filters )
    {
        split( filter.extensions, ";", [&] ( std::string_view&& ext )
        {
            assert( ext.starts_with( "*." ) );
            if ( ext != "*.*" )
                [fileTypes addObject:toNSString( ext.substr( 2 ) )];
            return false;
        } );
    }
    return fileTypes;
}

constexpr int cFileFormatPopupTag = 1;

NSView* createAccessoryView( const IOFilters& filters )
{
    auto* label = [NSTextField labelWithString:@"Format:"];
    label.textColor = NSColor.secondaryLabelColor;
    label.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];

    auto* popup = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    for ( const auto& filter : filters )
        [popup addItemWithTitle:toNSString( filter.name )];
    [popup setTag:cFileFormatPopupTag];

    auto* view = [[NSView alloc] initWithFrame:NSZeroRect];
    [view addSubview:label];
    [view addSubview:popup];

    // compute widget layout
    [label sizeToFit];
    [popup sizeToFit];
    const auto labelSize = [label frame].size;
    const auto popupSize = [popup frame].size;

    // see: https://stackoverflow.com/a/71276393
    label.translatesAutoresizingMaskIntoConstraints = NO;
    popup.translatesAutoresizingMaskIntoConstraints = NO;

    constexpr auto cPadding = 8.;
    constexpr auto cSpacing = 4.;
    [NSLayoutConstraint activateConstraints:@[
        [label.bottomAnchor constraintEqualToAnchor:view.bottomAnchor constant:-cPadding],
        [label.leadingAnchor constraintEqualToAnchor:view.leadingAnchor constant:+cPadding],
        [label.widthAnchor constraintEqualToConstant:labelSize.width],

        [popup.leadingAnchor constraintEqualToAnchor:label.trailingAnchor constant:cSpacing],
        [popup.firstBaselineAnchor constraintEqualToAnchor:label.firstBaselineAnchor],

        [popup.topAnchor constraintEqualToAnchor:view.topAnchor constant:+cPadding],
        [popup.trailingAnchor constraintEqualToAnchor:view.trailingAnchor constant:-cPadding],
        [popup.widthAnchor constraintEqualToConstant:popupSize.width],
    ]];

    return view;
}

} // namespace

/// helper class to update allowed file types on user change
@interface FileFormatPickerListener : NSObject
{
@private
    NSSavePanel* dialog_;
    MR::IOFilters filters_;
}
- (instancetype)initWithDialog:(NSSavePanel*)dialog filters:(MR::IOFilters)filters;
- (void)popupAction:(id)sender;
@end

@implementation FileFormatPickerListener

- (instancetype)initWithDialog:(NSSavePanel*)dialog filters:(MR::IOFilters)filters
{
    if ( ( self = [super init] ) )
    {
        dialog_ = dialog;
        filters_ = filters;
    }
    return self;
}

- (void)popupAction:(id)sender
{
    auto index = [sender indexOfSelectedItem];
    assert( index < filters_.size() );
    auto* fileTypes = makeFileTypes( { filters_[index] } );
    if ( [fileTypes count] == 0 )
        fileTypes = makeFileTypes( filters_ );
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [dialog_ setAllowedFileTypes:fileTypes];
#pragma clang diagnostic pop
}

@end

namespace MR::detail
{

std::vector<std::filesystem::path> runCocoaFileDialog( const MR::FileDialog::Parameters& params )
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

    const auto currentFolder = params.baseFolder.empty() ?
        MR::FileDialog::getLastUsedDir() : utf8string( params.baseFolder );
    if ( !currentFolder.empty() )
        [dialog setDirectoryURL:toNSURL( currentFolder )];

    if ( !params.fileName.empty() )
        [dialog setNameFieldStringValue:toNSString( params.fileName )];

    if ( !params.folderDialog && !params.filters.empty() )
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        [dialog setAllowedFileTypes:makeFileTypes( params.filters )];
#pragma clang diagnostic pop

        // add file format picker
        auto* accessoryView = createAccessoryView( params.filters );
        dialog.accessoryView = accessoryView;

        auto* popup = (NSPopUpButton*)[accessoryView viewWithTag:cFileFormatPopupTag];
        auto* pickerListener = [[FileFormatPickerListener alloc] initWithDialog:dialog filters:params.filters];
        popup.target = pickerListener;
        popup.action = @selector( popupAction: );
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
    if ( results.empty() )
        return results;

    const auto currentDir = params.folderDialog ? results.front() : results.front().parent_path();
    [[maybe_unused]] std::error_code ec;
    assert( is_directory( currentDir, ec ) );
    MR::FileDialog::setLastUsedDir( utf8string( currentDir ) );

    return results;
}

} // namespace MR::detail
