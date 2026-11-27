#pragma once

#include "MRViewerFwd.h"
#include "MRScannerBackend.h"

namespace MR
{

/// Registers a scanner backend for MeshLib's Scanner tool on module loading, and unregisters it on module unloading,
/// closing the tool first if it works with that backend; the tool lists the backends in registration order,
/// opens the first one unless another is saved, and is unavailable while no backend is registered
class ScannerBackendAdder
{
public:
    MRVIEWER_API explicit ScannerBackendAdder( std::unique_ptr<IScannerBackend> backend );
    MRVIEWER_API ~ScannerBackendAdder();

private:
    const IScannerBackend* backend_;
};

} // namespace MR
