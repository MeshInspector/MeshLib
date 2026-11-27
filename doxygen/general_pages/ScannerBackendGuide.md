# How to Connect a 3D Scanner {#ScannerBackendGuide}

[TOC]

# Overview

This page is for makers and integrators of 3D scanners and depth cameras who have a C++ SDK and want their device to work in the MeshLib viewer and in MeshInspector.

Both applications include a generic **Scanner** tool. It does everything that is the same for every scanner:
- the tool window, with a scanner selector, tabs, and Capture, Connect and Live buttons;
- widgets for the scanner's parameters, plus Reset and Save defaults;
- the 2D image preview and the live 3D preview;
- clipping scans with a plane;
- turntable rotation of earlier scans;
- ICP registration of each new scan to the earlier ones;
- adding every scan to the scene with undo.

The scanner-specific part is a **backend**. It implements one C++ interface, [MR::IScannerBackend](\ref MR::IScannerBackend) from `MRViewer/MRScannerBackend.h`, on top of your SDK. Build the backend as a plugin module and register it, and the Scanner tool lists your scanner by name. The backend does not use ImGui: it describes its parameters and actions as JSON data, and the tool draws them.

> [!NOTE]
> The Scanner tool and `IScannerBackend` are new. Build your plugin against a MeshLib SDK whose `MRViewer/MRScannerBackend.h` exists, and deploy it to a host of exactly that version. See [Version compatibility](\ref HowtoAddPluginOverview_VersionCompatibility).

# Using the Scanner tool {#ScannerBackendGuide_Using}

1. The tool is in the **Experimental** ribbon tab, group **Scanners**. That tab is visible after **Settings → Application → Show Experimental Features** is checked (the choice is saved), or when the application is started with `-develop`.
2. The **Scanner** button is disabled, with the tooltip "No scanner backend is registered", until a plugin registers a backend.
3. When the tool opens, it selects the backend chosen last time, or the first one by name. It then loads that backend's SDK runtime from the saved folder and connects to the scanner. To skip loading, for example when an SDK crashes the application, hold **Shift** while clicking the tool's button.
4. While the runtime is not loaded, the window shows why, the backend's hint, and the SDK folder with a **...** button to pick another one. Picking a folder loads the runtime from it and connects.
5. With the runtime loaded, the window has these tabs:
   - **Parameters**: the backend's parameters, then **Reset** and **Save defaults**;
   - **Clip Plane**: a plane drawn in the viewport. **Apply** clips the scans taken so far, and every new scan is clipped by it;
   - **Rot/Reg**: turntable rotation and automatic ICP registration;
   - **Current Scan**: the backend's own actions and a preview of its 2D images. This tab appears only if the backend has actions.
6. Below the tabs is the capture block:
   - **Capture** (or Space) takes a scan; it is enabled while the backend is ready;
   - **Live**, **Live + 3D**, **Stop** and **Capture frame** control the live preview, if the backend supports it;
   - **Connect** and **Disconnect**, unless the backend hides them;
   - the message of the last failure.
7. Each scan becomes a new scene object named `Scan0`, `Scan1` and so on. The first scan also fits the camera to the scene.

# Writing a backend {#ScannerBackendGuide_Writing}

## The interface {#ScannerBackendGuide_Interface}

The reference of every method is on the [MR::IScannerBackend](\ref MR::IScannerBackend) page. This table shows when the Scanner tool calls each one.

| Method | Called by the host | Your backend |
|---|---|---|
| `name()` | every frame | Returns a short, unique and stable name, e.g. `"MyScanner"`. The scanner selector shows it, the saved selection refers to it, and `toLower( name() )` is the key of your settings. |
| `runtimeLoaded()` | every frame | Tells whether the SDK runtime is loaded. |
| `isReady()` | every frame | Tells whether `capture()` can run now, e.g. the scanner is connected. It enables Capture and switches Connect to Disconnect. |
| `runtimeDir()` | when the tool opens; on the "not loaded" page | Returns the SDK folder used last time. It is empty at first. |
| `loadRuntime( dir )` | when the tool opens; when the user picks a folder | Loads the SDK from `dir` and saves `dir` so that `runtimeDir()` returns it. An empty `dir` should try the default library search. |
| `unloadRuntime()` | when the tool closes; when another backend is selected | Stops the live preview, disconnects, and releases every SDK object. It is also called when the runtime was never loaded. |
| `runtimeHint()` | on the "not loaded" page | Tells the user where to find the SDK folder. |
| `connect()` / `disconnect()` | after `loadRuntime()`; Connect and Disconnect buttons | Finds the scanner and connects to it. `connect()` does nothing when already connected. |
| `capabilities()` | every frame | Returns which optional parts of the window to show, as `ScannerCapabilities`: `livePreview` (default false) for the live preview buttons, `connectButtons` (default true) for Connect and Disconnect, and `actions` (default true) for the Current Scan tab. |
| `paramSchema()` | every frame while the Parameters or Current Scan tab is open | Returns the parameters and actions as JSON; see [Parameters and actions](\ref ScannerBackendGuide_Schema). |
| `setParam( key, value )` | when the user changes a parameter | Applies the new value. |
| `runAction( key, arg )` | when the user clicks an action button | Runs the action. Returns the scan it produced, or null when it produced none. |
| `capture()` | Capture button or Space | Takes one scan with the current parameters and returns it. |
| `previewImages()` | after a capture or an action | Returns 2D images of the current or last capture for the Current Scan tab. |
| `setLiveFrameCallback`, `startLivePreview`, `stopLivePreview`, `livePreviewActive`, `captureFrame` | live preview buttons; when the tool opens and closes | See [Live preview](\ref ScannerBackendGuide_Live). Needed only with `capabilities().livePreview`. |
| `upDirection()` | after each scan | Returns the up direction in the scanner's frame, used to turn the camera toward a new scan. The default is −Y, as in most camera frames. |
| `loadSettings()` / `saveSettings()` / `resetSettings()` | tool open / tool close, backend switch and Save defaults / Reset | See [Settings](\ref ScannerBackendGuide_Settings). |

The methods without a default implementation must be overridden.

## Rules every backend follows {#ScannerBackendGuide_Rules}

- **The module must load without the SDK.** The backend object is created when your plugin module loads, before the user has picked an SDK folder. Its constructor must not call the SDK, and nothing may touch the SDK before `loadRuntime()`. The tested way to do this is on Windows:
  - link the SDK with `/DELAYLOAD:<sdk>.dll` and `delayimp.lib`;
  - in `loadRuntime( dir )`, call `SetDllDirectoryW( dir )`, `LoadLibrary` the SDK, and resolve its imports with `__HrLoadAllImportsForDll`, then reset the directory with `SetDllDirectoryW( nullptr )`;
  - do this inside `__try` / `__except`, so that an incompatible SDK version is reported as a failure instead of crashing the application.
- **Everything runs on the GUI thread**, except the live frame sink. `capture()` and `runAction()` block the UI while they run.
- **Per-frame methods must be cheap and must not block.** These are `name()`, `runtimeLoaded()`, `isReady()`, `capabilities()`, `livePreviewActive()` and `paramSchema()`. Return cached state; don't query the device from them.
- **Failures are text.** Every method that can fail returns `MR::Expected`, and its error string is shown in the tool window. Failures of `capture()` and `runAction()` are also written to the log. Write the texts in English and mark literals with `_t( "..." )` so they can be translated (see \ref I18nGuide).

## Parameters and actions {#ScannerBackendGuide_Schema}

`paramSchema()` describes the backend's UI as JSON. The host calls it again every frame, so it can change with the state; for example, a parameter can hide or disable another one.

```json
{
  "groups": [
    { "title": "Exposure", "params": [ <param>, ... ] }
  ],
  "actionGroups": [
    { "title": "Raw data", "status": "Frames held: {}", "statusArg": "3", "actions": [ <action>, ... ] }
  ]
}
```

The `groups` are drawn on the Parameters tab. The `actionGroups` are drawn on the Current Scan tab, three buttons in a row. Titles are optional. In a `status`, `{}` is replaced with `statusArg`; write a literal brace as `{{` or `}}`.

A `<param>` has `key`, `type`, `label` and `value`; the helper `MR::scannerParam( key, label, type, value )` makes one. These are the types, and the JSON values that `setParam` receives for them:

| `type` | Widget | Value passed to `setParam` | Optional fields |
|---|---|---|---|
| `bool` | checkbox | bool | |
| `float` | drag field | double | `min`, `max`, `step`, `precision`, `unit` (only `"ms"`) |
| `int` | drag field | int | `min`, `max`, `step` |
| `enum` | combo box | int, the index in `options` | `options`: the labels |
| `path` | file name with a **...** button | string, a UTF-8 path | `filters`, e.g. `"*.cfg;*.yml"` |
| `info` | read-only text `label: value` | never called | |

Every parameter except `info` also takes `enabled` (false disables the widget) and `tooltip`.

An `<action>` has `key` and `label`, and these optional fields:
- `tooltip` and `enabled`;
- `filters`: the button first asks for a file to open, and passes its UTF-8 path as the `arg` of `runAction`. Otherwise `arg` is null. With `save: true`, it asks for a file to save instead;
- `advanced: true`: the button is shown only while Ctrl+Shift (Cmd+Shift on macOS) is held;
- `experimental: true`: the button is shown only with experimental features on.

Keys must be unique within the schema. Labels may repeat, because widget IDs are `label##key`.

## Returning scans {#ScannerBackendGuide_Scans}

`capture()` and `runAction()` return a new `MR::ObjectPoints` (a point cloud, preferably with normals) or `MR::ObjectMesh`, with no parent and in the scanner's coordinate frame. Don't name it or add it to the scene. The host:
1. names it `ScanN` and selects it;
2. clips it with the Clip Plane, if one is set;
3. turns the earlier scans by the turntable angle, if Auto rotate is on;
4. registers it to the earlier scans with ICP, if Auto register is on. This replaces its transformation;
5. adds it to the scene as one undo step, and turns the camera toward it.

## Live preview {#ScannerBackendGuide_Live}

When `capabilities().livePreview` is true, the capture block shows **Live**, **Live + 3D** (which call `startLivePreview( false / true )`), **Stop** and **Capture frame**.

- When the tool opens, the host passes a sink to `setLiveFrameCallback`. Send every frame to it as a `ScannerLiveFrame`, holding 2D images and, for Live + 3D, the frame's points. An empty image keeps the previous image in its slot. When the tool closes, the host sets an empty callback; stop calling the old one.
- The sink may be called from any SDK thread, and also from inside `captureFrame()`. It only stores the frame and queues a GUI update, so it returns quickly.
- `stopLivePreview()` may wait for your SDK thread to finish. While it waits, your SDK thread must never wait for the GUI thread; that would deadlock.
- The live points are shown as a temporary point cloud; they are not a scan. A scan is taken only by Capture or by an action.

## Settings {#ScannerBackendGuide_Settings}

A backend keeps all of its settings, including the SDK folder, in the application's config file, in the JSON object `toLower( name() )`. Use `MR::Config::instance().getJsonValue( key )` and `setJsonValue( key, json )`. The host keeps its own Rot/Reg settings in **the same object**. Their fields are `rotate`, `rotateDegrees`, `rotateOrg`, `rotateOrgSet`, `rotateAxis`, `registration` and `ransac`, so:
- don't use these field names;
- in `saveSettings()`, read the object, change your fields, and write it back. Never replace the whole object.

The host calls `loadSettings()` when the tool opens or the backend is selected, before `loadRuntime()`. It calls `saveSettings()` when the tool closes, when another backend is selected, and on **Save defaults**, and `resetSettings()` on **Reset**. A backend may also save a setting as soon as it changes.

# A minimal backend {#ScannerBackendGuide_Example}

This sample needs no hardware: its `capture()` returns a flat grid of points instead of SDK data. Replace the commented places with calls to your SDK.

```cpp
#include "MRViewer/MRScannerBackend.h"
#include "MRMesh/MRConfig.h"
#include "MRMesh/MRObjectPoints.h"

namespace
{

using namespace MR;

constexpr const char* cConfigKey = "myscanner"; // toLower( name() )
constexpr float cDefaultExposure = 10.0f;

class MyScannerBackend : public IScannerBackend
{
public:
    std::string name() const override { return "MyScanner"; }
    bool runtimeLoaded() const override { return loaded_; }
    bool isReady() const override { return connected_; }

    std::string runtimeDir() const override { return sdkDir_; }
    Expected<void> loadRuntime( const std::string& dir ) override
    {
        sdkDir_ = dir;
        saveSettings(); // runtimeDir() returns the picked folder next time
        // load the SDK from dir here; on failure: return unexpected( _t( "Loading MyScanner SDK failed" ) );
        loaded_ = true;
        return {};
    }
    void unloadRuntime() override
    {
        disconnect();
        // release all SDK objects here
        loaded_ = false;
    }
    std::string runtimeHint() const override { return _t( "Select the MyScanner SDK binary folder" ); }

    Expected<void> connect() override
    {
        // find and open the scanner here; on failure: return unexpected( _t( "No MyScanner found" ) );
        connected_ = true;
        return {};
    }
    void disconnect() override { connected_ = false; }

    ScannerCapabilities capabilities() const override { return { .actions = false }; }

    Json::Value paramSchema() const override
    {
        Json::Value exposure = scannerParam( "exposure", _t( "Exposure" ), "float", double( exposure_ ) );
        exposure["min"] = 1.0;
        exposure["max"] = 100.0;
        exposure["unit"] = "ms";
        Json::Value schema;
        schema["groups"][0]["params"].append( exposure );
        return schema;
    }
    void setParam( const std::string& key, const Json::Value& value ) override
    {
        if ( key == "exposure" )
            exposure_ = value.asFloat(); // pass it to the SDK here
    }

    Expected<std::shared_ptr<VisualObject>> runAction( const std::string& key, const Json::Value& ) override
    {
        return unexpected( "Unknown action: " + key );
    }

    Expected<std::shared_ptr<VisualObject>> capture() override
    {
        // take a scan with the SDK here and copy its points, in the scanner's frame
        auto cloud = std::make_shared<PointCloud>();
        for ( int y = 0; y < 100; ++y )
            for ( int x = 0; x < 100; ++x )
                cloud->points.push_back( Vector3f( float( x ), float( y ), 0.0f ) );
        cloud->validPoints.resize( cloud->points.size(), true );
        auto obj = std::make_shared<ObjectPoints>();
        obj->setPointCloud( std::move( cloud ) );
        return obj;
    }

    void loadSettings() override
    {
        const Json::Value json = Config::instance().getJsonValue( cConfigKey );
        sdkDir_ = json.get( "sdkPath", "" ).asString();
        exposure_ = json.get( "exposure", cDefaultExposure ).asFloat();
    }
    void saveSettings() override
    {
        Json::Value json = Config::instance().getJsonValue( cConfigKey ); // keeps the host's fields
        json["sdkPath"] = sdkDir_;
        json["exposure"] = exposure_;
        Config::instance().setJsonValue( cConfigKey, json );
    }
    void resetSettings() override
    {
        exposure_ = cDefaultExposure; // the SDK folder stays
        saveSettings();
    }

private:
    std::string sdkDir_;
    float exposure_ = cDefaultExposure;
    bool loaded_ = false;
    bool connected_ = false;
};

// registered when the module loads, and unregistered when the application closes
ScannerBackendAdder sMyScannerBackend( std::make_unique<MyScannerBackend>() );

} // namespace
```

One module may register several backends, one `ScannerBackendAdder` each. They are listed by name.

# Building and installing {#ScannerBackendGuide_Building}

1. Start from the [C++ example plugin](\ref ExamplePluginOverview), which already builds a module that links `MRMesh` and `MRViewer`. Replace `MyPlugin.cpp` with your backend.
2. The Scanner tool is already registered and placed in the ribbon by MeshLib, so your module adds no ribbon items. Remove `MyPlugin.items.json` and the `resource` folder from the project. Keep a `*.ui.json` file that only tells the host to load your library. Its `LibName` is the library file name without the `lib` prefix and the extension:
```json
{
	"Order": 999,
	"LibName": "MyScannerPlugin",
	"Tabs": []
}
```
3. On Windows, add the delay-load linker options from [Rules every backend follows](\ref ScannerBackendGuide_Rules), and build with the same configuration as the host (Release for an installed MeshInspector).
4. Install the library and the `*.ui.json` file as described in \ref HowtoAddPluginOverview, and check the log for `Load library MyScannerPlugin was successful`. The SDK itself is not copied into the application folder: the user picks its folder in the Scanner tool, and that folder is saved.

# Troubleshooting {#ScannerBackendGuide_Troubleshooting}

| Symptom | Likely cause |
|---|---|
| The Scanner button is disabled: "No scanner backend is registered" | The module did not load: look for its `LibName` in the log. Or the `ScannerBackendAdder` is missing. |
| The plugin module fails to load when the SDK is not on `PATH` | The SDK is linked normally; delay-load it and load it in `loadRuntime()`. |
| The application crashes at startup | The backend's constructor or a static object calls the SDK. |
| The UI stutters while the tool is open | A per-frame method (`isReady()`, `paramSchema()` and so on) talks to the device. |
| The application hangs on Stop or when the tool closes | The SDK thread that calls the live sink waits for the GUI thread. |
| Rot/Reg settings are lost after closing the tool | `saveSettings()` replaces the whole Config object instead of updating its own fields. |
