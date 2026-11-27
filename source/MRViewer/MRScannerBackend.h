#pragma once

#include "exports.h"
#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRExpected.h>
#include <MRMesh/MRImage.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRVector3.h>
#include "MRPch/MRJson.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace MR
{

// Optional UI regions, queried every frame.
struct ScannerCapabilities
{
    bool livePreview = false;
    bool connectButtons = true; // off where connecting is not a user step, e.g. for a file camera
    bool actions = true; // the Current Scan tab with the schema's action groups; off when it has none
};

// One live-preview frame: 2D images (an empty one keeps its slot's previous image) and, if streamed, points.
struct ScannerLiveFrame
{
    std::vector<Image> images;
    std::optional<PointCloud> points;
};

// A scanner SDK backend without ImGui/MRViewer dependencies; the host and the generic UI use only this.
// Called on the GUI thread; per-frame getters must not block; nothing touches the delay-loaded SDK
// before loadRuntime(). Failures are English text (literals marked _t) in the returned Expected.
class IScannerBackend
{
public:
    virtual ~IScannerBackend() = default;

    // The scanner's name, e.g. "Polyga".
    virtual std::string name() const = 0;
    virtual bool runtimeLoaded() const = 0;
    virtual bool isReady() const = 0; // can capture(), e.g. connected

    // The SDK runtime: the persisted folder, loading from it (then the host calls connect()),
    // releasing every SDK object (also when never loaded), and a hint while not loaded.
    virtual std::string runtimeDir() const = 0;
    virtual Expected<void> loadRuntime( const std::string& dir ) = 0;
    virtual void unloadRuntime() = 0;
    virtual std::string runtimeHint() const = 0;

    // Finds the scanner if needed and connects; does nothing when connected.
    virtual Expected<void> connect() = 0;
    virtual void disconnect() = 0;

    virtual ScannerCapabilities capabilities() const = 0;

    // 2D images of the current or last capture.
    virtual std::vector<Image> previewImages() const { return {}; }
    // Live preview: the sink may be called on any thread, also from captureFrame(); it must not wait for
    // the GUI thread, which stopLivePreview() may block.
    virtual void setLiveFrameCallback( std::function<void( ScannerLiveFrame&& )> ) {}
    virtual void startLivePreview( bool withPoints ) { (void)withPoints; }
    virtual void stopLivePreview() {}
    virtual bool livePreviewActive() const { return false; }
    virtual Expected<void> captureFrame() { return {}; }

    // The UI as data, rebuilt on every call:
    //   { "groups": [ { "title"?, "params": [`<param>`...] } ],
    //     "actionGroups": [ { "title"?, "status"?, "statusArg"?, "actions": [`<action>`...] } ] }
    // `<param>`: key, type (float|int|bool|enum|info|path), label, value; for a path also filters ("*.a;*.b",
    //   named by the label; the value is a UTF-8 path picked from such files); optional: enabled, tooltip,
    //   min, max, step, unit ("ms"), precision, options (enum labels; the value is the index).
    // `<action>`: key, label; optional: tooltip, enabled, filters (pick a file, passed as a UTF-8 path),
    //   save (pick a file to write), advanced (shown with Ctrl+Shift), experimental.
    // Texts are English, marked _t; "{}" in a status is replaced by statusArg. Widget IDs are `label##key`
    //   (a path: `##key` and `...##key`).
    virtual Json::Value paramSchema() const = 0;
    virtual void setParam( const std::string& key, const Json::Value& value ) = 0;

    // Runs a schema action; returns the produced scan, or null.
    virtual Expected<std::shared_ptr<VisualObject>> runAction( const std::string& key, const Json::Value& arg ) = 0;

    // One scan with the current parameters.
    virtual Expected<std::shared_ptr<VisualObject>> capture() = 0;

    // The up direction of the scanner's frame, to orient the camera to a new scan.
    virtual Vector3f upDirection() const { return Vector3f::minusY(); }

    // All settings, in the Config object toLower( name() ) shared with the host's settings (rotate*, registration),
    // so saveSettings() updates its own fields there, never replaces the object. The host loads them once and saves
    // them whenever it closes the backend.
    virtual void loadSettings() = 0;
    virtual void saveSettings() = 0;
    virtual void resetSettings() = 0;
};

// A schema `<param>` with its required fields.
inline Json::Value scannerParam( std::string key, std::string label, std::string type, Json::Value value )
{
    Json::Value p;
    p["key"] = std::move( key );
    p["label"] = std::move( label );
    p["type"] = std::move( type );
    p["value"] = std::move( value );
    return p;
}

/// Registers a scanner backend for the Scanner tool on module loading, and unregisters it on module unloading
/// unless unregisterScannerBackends() did; the tool lists the backends in registration order,
/// opens the first one unless another is saved, and is unavailable while no backend is registered
class ScannerBackendAdder
{
public:
    MRVIEWER_API explicit ScannerBackendAdder( std::unique_ptr<IScannerBackend> backend );
    MRVIEWER_API ~ScannerBackendAdder();

private:
    const IScannerBackend* backend_;
};

/// The registered scanner backends, in registration order
MRVIEWER_API const std::vector<std::unique_ptr<IScannerBackend>>& getScannerBackends();

/// Destroys all registered scanner backends; the Scanner tool calls it on viewer shutdown, after closing,
/// so the backends are destroyed before the modules implementing them unload
MRVIEWER_API void unregisterScannerBackends();

} // namespace MR
