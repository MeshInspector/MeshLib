#include "MRScannerBackend.h"

#include <algorithm>

namespace MR
{

namespace
{

std::vector<std::unique_ptr<IScannerBackend>>& scannerBackends()
{
    static std::vector<std::unique_ptr<IScannerBackend>> backends;
    return backends;
}

} // namespace

ScannerBackendAdder::ScannerBackendAdder( std::unique_ptr<IScannerBackend> backend ) :
    backend_( backend.get() )
{
    // sorted by name, since the order of static initialization in different translation units is unspecified
    auto& backends = scannerBackends();
    const auto pos = std::upper_bound( backends.begin(), backends.end(), backend->name(),
        [] ( const std::string& name, const auto& b ) { return name < b->name(); } );
    backends.insert( pos, std::move( backend ) );
}

ScannerBackendAdder::~ScannerBackendAdder()
{
    std::erase_if( scannerBackends(), [this] ( const auto& b ) { return b.get() == backend_; } );
}

const std::vector<std::unique_ptr<IScannerBackend>>& getScannerBackends()
{
    return scannerBackends();
}

void unregisterScannerBackends()
{
    scannerBackends().clear();
}

} // namespace MR
