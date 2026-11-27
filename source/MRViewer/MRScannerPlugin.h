#pragma once

#include "MRViewerFwd.h"
#include "MRScannerBackend.h"

namespace MR
{

/// Registers a scanner backend for the Scanner tool on module loading, and unregisters it on module unloading;
/// the tool lists the backends in registration order, and opens the first one unless another is saved
class ScannerBackendAdder
{
public:
    MRVIEWER_API explicit ScannerBackendAdder( std::unique_ptr<IScannerBackend> backend );
    MRVIEWER_API ~ScannerBackendAdder();

private:
    const IScannerBackend* backend_;
};

/// Creates the Scanner tool: it lists the registered backends, communicates with the selected one only through
/// IScannerBackend, renders its parameter schema, and processes the scans (clip / rotate / register, scene insertion);
/// it is unavailable while no backend is registered
MRVIEWER_API std::shared_ptr<RibbonMenuItem> makeScannerPlugin();

} // namespace MR
