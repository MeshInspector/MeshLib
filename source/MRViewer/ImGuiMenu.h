#pragma  once
// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2018 Jérémie Dumas <jeremie.dumas@ens-lyon.org>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

////////////////////////////////////////////////////////////////////////////////
#include "MRMeshViewer.h"
#include "MRMeshViewerPlugin.h"
#include "MRViewerEventsListener.h"
////////////////////////////////////////////////////////////////////////////////

// Forward declarations
struct ImGuiContext;


namespace MR
{

class ShortcutManager;

class MRVIEWER_CLASS ImGuiMenu : public MR::ViewerPlugin, 
    public MultiListener<
    MouseDownListener, MouseMoveListener, MouseUpListener, MouseScrollListener,
    CharPressedListener, KeyDownListener, KeyUpListener, KeyRepeatListener,
    PreDrawListener, PostDrawListener,
    PostResizeListener, PostRescaleListener>
{
    using ImGuiMenuMultiListener = MultiListener<
        MouseDownListener, MouseMoveListener, MouseUpListener, MouseScrollListener,
        CharPressedListener, KeyDownListener, KeyUpListener, KeyRepeatListener,
        PreDrawListener, PostDrawListener,
        PostResizeListener>;
protected:
  // Hidpi scaling to be used for text rendering.
  float hidpi_scaling_;

  // Ratio between the framebuffer size and the window size.
  // May be different from the hipdi scaling!
  float pixel_ratio_;

  // ImGui Context
  ImGuiContext * context_ = nullptr;

  // if true, then pre_draw will start from polling glfw events
  bool pollEventsInPreDraw = false; // be careful here with true, this can cause infinite recurse 

  bool showStatistics_{ false };
  long long frameTimeMillisecThreshold_{ 25 };
  bool show_rename_modal_{ false };
  std::string renameBuffer_;
  std::string storedError_;

public:
  MRVIEWER_API virtual void init(MR::Viewer *_viewer) override;

  // inits glfw and glsl backend
  MRVIEWER_API virtual void initBackend();

  MRVIEWER_API virtual void load_font(int font_size = 13);
  MRVIEWER_API virtual void reload_font(int font_size = 13);

  MRVIEWER_API virtual void shutdown() override;

  // Draw menu
  MRVIEWER_API virtual void draw_menu();

  MRVIEWER_API void draw_helpers();

  // Can be overwritten by `callback_draw_viewer_window`
  MRVIEWER_API virtual void draw_viewer_window();

  // Can be overwritten by `callback_draw_viewer_menu`
  //virtual void draw_viewer_menu();

  // Can be overwritten by `callback_draw_custom_window`
  virtual void draw_custom_window() {}

  // Easy-to-customize callbacks
  std::function<void(void)> callback_draw_viewer_window;
  std::function<void(void)> callback_draw_viewer_menu;
  std::function<void(void)> callback_draw_custom_window;

  void draw_labels_window();

  void draw_labels( const VisualObject& obj );

  MRVIEWER_API void draw_text(
      const Viewport& viewport,
      const Vector3f& pos,
      const Vector3f& normal,
      const std::string& text,
      const Color& color,
      bool clipByViewport,
      bool useStaticMatrix = false ); // for basis axis

  MRVIEWER_API float pixel_ratio();

  MRVIEWER_API float hidpi_scaling();

  MRVIEWER_API float menu_scaling() const;

  MRVIEWER_API ImGuiContext* getCurrentContext() const;

protected:
    
    std::shared_ptr<ShortcutManager> shortcutManager_;
    bool capturedMouse_{ false };
    // Mouse IO
    MRVIEWER_API virtual bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;
    MRVIEWER_API virtual bool onMouseScroll_( float delta_y ) override;
    // Keyboard IO
    MRVIEWER_API virtual bool onCharPressed_( unsigned key, int modifiers ) override;
    MRVIEWER_API virtual bool onKeyDown_( int key, int modifiers ) override;
    MRVIEWER_API virtual bool onKeyUp_( int key, int modifiers ) override;
    MRVIEWER_API virtual bool onKeyRepeat_( int key, int modifiers ) override;
    // Render events
    MRVIEWER_API virtual void preDraw_() override;
    MRVIEWER_API virtual void postDraw_() override;
    // Scene events
    MRVIEWER_API virtual void postResize_( int width, int height ) override;
    MRVIEWER_API virtual void postRescale_( float x, float y) override;

    // This function reset ImGui style to current theme and scale it by menu_scaling
    // called in ImGuiMenu::postRescale_()
    MRVIEWER_API virtual void rescaleStyle_();

    // setup maximum good time for frame rendering (if rendering is slower it will become red in statistics window)
    MRVIEWER_API void setDrawTimeMillisecThreshold( long long maxGoodTimeMillisec );
};

} // end namespace
