#include "MRScannerBackend.h"

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
    scannerBackends().push_back( std::move( backend ) );
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
