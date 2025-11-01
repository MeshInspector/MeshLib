#pragma once

#include "config.h"
#include "exports.h"
#include <MRMesh/MRMeshFwd.h>
#include <functional>

struct ImFont;

namespace MR
{

/// Viewport size
using ViewportRectangle = Box2f;

enum class FitMode;
struct BaseFitParams;
struct FitDataParams;
struct FitBoxParams;
struct FileLoadOptions;

enum class MouseButton;
enum class MouseMode;

class AlphaSortGL;
class ColorTheme;
class ImGuiImage;
class ImGuiMenu;
class IViewerSettingsManager;
class IDragDropHandler;
class FrameCounter;
class MarkedVoxelSlice;
class Palette;
class RecentFilesStore;
class ScopeHistory;
class SelectScreenLasso;
class SceneTextureGL;
class SpaceMouseHandler;
class SpaceMouseHandlerHidapi;
class SplashWindow;
class StateBasePlugin;
class Toolbar;
class ViewerPlugin;
class ViewerSettingsManager;
class ViewerSetup;
class ViewerTitle;
class Viewer;
struct ViewerSignals;
struct LaunchParams;
class ViewerEventQueue;
class ViewportGlobalBasis;
class Viewport;
class RibbonMenu;
class RibbonMenuItem;
class RibbonFontManager;
class ShortcutManager;
struct ShortcutKey;
enum class ShortcutCategory : char;

class DirectionWidget;
class PlaneWidget;

class TouchpadController;
struct TouchpadParameters;
class SpaceMouseController;
struct SpaceMouseParameters;
class TouchesController;
class MouseController;
struct PointInAllSpaces;
class CornerControllerObject;

template<typename ...Connectables>
class StateListenerPlugin;
using StatePlugin = StateListenerPlugin<>;

class HistoryStore;

using ViewerEventCallback = std::function<void()>;

class MRVIEWER_CLASS WebRequest;

struct PointOnObject;

using ObjAndPick = std::pair<std::shared_ptr<MR::VisualObject>, MR::PointOnObject>;
using ConstObjAndPick = std::pair<std::shared_ptr<const MR::VisualObject>, MR::PointOnObject>;

using RequirementsFunction = std::function<std::string( const std::shared_ptr<RibbonMenuItem>& )>;

using FontAndSize = std::pair<ImFont*, float>;

// this is needed as far as MAKE_SLOT cannot be used with movable classes
#define MR_DELETE_MOVE(ClassName)\
ClassName(ClassName&&)noexcept = delete;\
ClassName& operator=(ClassName&&)noexcept = delete

#define MR_ADD_CTOR_DELETE_MOVE(ClassName)\
ClassName()=default;\
ClassName(ClassName&&)noexcept = delete;\
ClassName& operator=(ClassName&&)noexcept = delete

} //namespace MR
