#include "MRScannerBackend.h"

namespace MR
{

ScannerBackendRegistry& getScannerBackendRegistry()
{
    static ScannerBackendRegistry registry;
    return registry;
}

ScannerBackendAdder::ScannerBackendAdder( std::unique_ptr<IScannerBackend> backend ) :
    backend_( backend.get() )
{
    getScannerBackendRegistry().backends.push_back( std::move( backend ) );
}

ScannerBackendAdder::~ScannerBackendAdder()
{
    auto& registry = getScannerBackendRegistry();
    if ( registry.onUnregister )
        registry.onUnregister( *backend_ );
    std::erase_if( registry.backends, [this] ( const auto& b ) { return b.get() == backend_; } );
}

} // namespace MR
