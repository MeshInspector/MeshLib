#include "MRLocaleMacos.h"

#include <Foundation/Foundation.h>

namespace MR::Locale::detail
{

std::vector<std::string> getMacosLocales()
{
    std::vector<std::string> results;
    // enable garbage collector
    auto* pool = [[NSAutoreleasePool alloc] init];
    for ( NSString* lang in [NSLocale preferredLanguages] )
        results.emplace_back( lang.UTF8String );
    [pool release];
    return results;
}

} // namespace MR::Locale::detail
