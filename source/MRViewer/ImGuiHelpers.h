#pragma once
// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2018 Jérémie Dumas <jeremie.dumas@ens-lyon.org>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "MRMesh/MRFlagOperators.h"
#include "exports.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRColor.h"
#include "MRViewer/MRViewerFwd.h"
#include "MRViewer/MRUnits.h"
#include "MRViewer/MRImGui.h"
#include <misc/cpp/imgui_stdlib.h>
#include <algorithm>
#include <functional>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include <optional>

// Extend ImGui by populating its namespace directly
//
// Code snippets taken from there:
// https://eliasdaler.github.io/using-imgui-with-sfml-pt2/
namespace ImGui
{

static auto vector_getter = [](void* vec, int idx, const char** out_text)
{
  auto& vector = *static_cast<std::vector<std::string>*>(vec);
  if (idx < 0 || idx >= static_cast<int>(vector.size())) { return false; }
  *out_text = vector.at(idx).c_str();
  return true;
};

inline bool Combo(const char* label, int* idx, const std::vector<std::string>& values)
{
  if (values.empty()) { return false; }
  return Combo(label, idx, vector_getter,
    const_cast<void *>( static_cast<const void*>(&values) ), (int)values.size());
}

inline bool Combo(const char* label, int* idx, std::function<const char *(int)> getter, int items_count)
{
  auto func = [](void* data, int i, const char** out_text) {
    auto &getter = *reinterpret_cast<std::function<const char *(int)> *>(data);
    const char *s = getter(i);
    if (s) { *out_text = s; return true; }
    else { return false; }
  };
  return Combo(label, idx, func, reinterpret_cast<void *>(&getter), items_count);
}

inline bool ListBox(const char* label, int* idx, const std::vector<std::string>& values)
{
  if (values.empty()) { return false; }
  return ListBox(label, idx, vector_getter,
    const_cast<void *>( static_cast<const void*>(&values) ), (int)values.size());
}

inline bool InputText(const char* label, std::string &str, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL)
{
  return ImGui::InputText( label, &str, flags, callback, user_data );
}


/// similar to ImGui::DragFloat but
/// 1) value on output is forced to be in [min,max] range;
/// 2) tooltip about Shift/Alt is shown when the item is active, and the valid range
MRVIEWER_API bool DragFloatValid( const char *label, float* value, float speed=1.0f,
                                float min = std::numeric_limits<float>::lowest(),
                                float max = std::numeric_limits<float>::max(),
                                const char* format = "%.3f", ImGuiSliderFlags flags = 0 );

/// similar to ImGui::DragFloatValid but use available line width range
MRVIEWER_API bool DragFloatValidLineWidth( const char* label, float* value );

struct MultiDragRes
{
    bool valueChanged = false; // any of N
    bool itemDeactivatedAfterEdit = false; //any of N
    explicit operator bool() const { return valueChanged; }
};

/// similar to ImGui::DragFloat2 - two drag-float controls in a row, but
/// 1) returns information whether an item was deactivated;
/// 2) calls DragFloatValid inside instead of DragFloat;
/// 3) permits showing tooltip for each item
MRVIEWER_API MultiDragRes DragFloatValid2( const char* label, float v[2], float v_speed = 1.0f,
    float min = std::numeric_limits<float>::lowest(),
    float max = std::numeric_limits<float>::max(),
    const char* format = "%.3f", ImGuiSliderFlags flags = 0,
    const char* ( *tooltips )[2] = nullptr );

/// similar to ImGui::DragFloat3 - three drag-float controls in a row, but
/// 1) returns information whether an item was deactivated;
/// 2) calls DragFloatValid inside instead of DragFloat;
/// 3) permits showing tooltip for each item
MRVIEWER_API MultiDragRes DragFloatValid3( const char * label, float v[3], float v_speed = 1.0f,
    float min = std::numeric_limits<float>::lowest(),
    float max = std::numeric_limits<float>::max(),
    const char* format = "%.3f", ImGuiSliderFlags flags = 0,
    const char* (*tooltips)[3] = nullptr );

/// similar to ImGui::DragInt but
/// 1) value on output is forced to be in [min,max] range;
/// 2) tooltip about Shift/Alt is shown when the item is active, and the valid range
MRVIEWER_API bool DragIntValid( const char *label, int* value, float speed = 1,
                                int min = std::numeric_limits<int>::lowest(),
                                int max = std::numeric_limits<int>::max(),
                                const char* format = "%d" );

/// similar to ImGui::DragInt3 - three drag-int controls in a row, but
/// 1) returns information whether an item was deactivated;
/// 2) calls DragIntValid inside instead of DragInt;
/// 3) permits showing tooltip for each item
MRVIEWER_API MultiDragRes DragIntValid3( const char* label, int v[3], float speed = 1,
                                int min = std::numeric_limits<int>::lowest(),
                                int max = std::numeric_limits<int>::max(),
                                const char* format = "%d",
                                const char* ( *tooltips )[3] = nullptr );

// similar to ImGui::InputInt but
// 1) value on output is forced to be in [min,max] range;
// 2) tooltip about valid range is shown when the item is active
MRVIEWER_API bool InputIntValid( const char* label, int* value, int min, int max,
    int step = 1, int step_fast = 100, ImGuiInputTextFlags flags = 0 );

// draw check-box that takes initial value from Getter and then saves the final value in Setter
template<typename Getter, typename Setter>
inline bool Checkbox(const char* label, Getter get, Setter set)
{
    bool value = get();
    bool ret = ImGui::Checkbox(label, &value);
    set(value);
    return ret;
}

/// helper structure for PlotCustomHistogram describing background grid line and label
struct HistogramGridLine
{
    /// value on the corresponding axis where the line and label are located
    float value{};
    /// label text
    std::string label;
    /// label tooltip
    std::string tooltip;
};

/// draws a histogram
/// \param selectedBarId (if not negative) the bar to highlight as selected
/// \param hoveredBarId (if not negative) the bar to highlight as hovered
MRVIEWER_API void PlotCustomHistogram( const char* str_id,
                                 std::function<float( int idx )> values_getter,
                                 std::function<void( int idx )> tooltip,
                                 std::function<void( int idx )> on_click,
                                 int values_count, int values_offset = 0,
                                 float scale_min = FLT_MAX, float scale_max = FLT_MAX,
                                 ImVec2 frame_size = ImVec2( 0, 0 ), int selectedBarId = -1, int hoveredBarId = -1,
                                 const std::vector<HistogramGridLine>& gridIndexes = {},
                                 const std::vector<HistogramGridLine>& gridValues = {} );

/// begin typical state plugin window
MRVIEWER_API bool BeginStatePlugin( const char* label, bool* open, float width );

/// Structure that contains parameters for State plugin window with custom style
struct CustomStatePluginWindowParameters
{
    // All fields those have explicit initializers, even if they have sane default constructors.
    // This makes it so that Clangd doens't warn when they aren't initialized in partial aggregate initialization.

    /// current collapsed state of window
    /// in/out parameter, owned outside of `BeginCustomStatePlugin` function
    bool* collapsed{ nullptr };
    /// window width (should be already scaled with menuScaling)
    float width{ 0.0f };
    /// window height, usually calculated internally (if value is zero)
    float height{ 0.0f };
    /// If false, will never show the scrollbar.
    bool allowScrollbar = true;
    /// start Position
    ImVec2* position{ nullptr };
    /// the position of the starting point of the window
    ImVec2 pivot{ 0.0f, 0.0f };
    /// menu scaling, needed to proper scaling of internal window parts
    float menuScaling{ 1.0f };
    /// window flags, ImGuiWindowFlags_NoScrollbar and ImGuiWindow_NoScrollingWithMouse are forced inside `BeginCustomStatePlugin` function
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize;
    /// outside owned parameter for windows with resize option
    ImVec2* changedSize{ nullptr };
    /// draw custom header items immediately after the caption
    std::function<void()> customHeaderFn = nullptr;
    /// reaction on press "Help" button
    std::function<void()> helpBtnFn = nullptr;
    /// if true esc button closes the plugin
    bool closeWithEscape{ true };
};

/// returns the position of the window that will be located at the bottom of the viewport
/// for a value pivot = ( 0.0f, 1.0f )
MRVIEWER_API ImVec2 GetDownPosition( const float width );

/// Calculate and return the height of the window title
MRVIEWER_API float GetTitleBarHeght( float menuScaling );

/// Load saved window position, if possible
/// \details if can't load - get \p position (if setted) or default (upper-right viewport corner)
/// see also \ref SaveWindowPosition
/// \param label window label
/// \param width window width
/// \param position (optional) preliminary window position
/// \return pair of the final position of the window and flag whether the position was loaded
MRVIEWER_API std::pair<ImVec2, bool> LoadSavedWindowPos( const char* label, ImGuiWindow* window, float width, const ImVec2* position = nullptr );
inline std::pair<ImVec2, bool> LoadSavedWindowPos( const char* label, float width, const ImVec2* position = nullptr )
{
    return LoadSavedWindowPos( label, FindWindowByName( label ), width, position );
}
/// Save window position
/// \details saved only if window exist
/// see also \ref LoadSavedWindowPos
MRVIEWER_API void SaveWindowPosition( const char* label, ImGuiWindow* window );
inline void SaveWindowPosition( const char* label )
{
    SaveWindowPosition( label, FindWindowByName( label ) );
}

/// Parameters drawing classic ImGui::Begin with loading / saving window position
struct SavedWindowPosParams
{
    /// window size
    ImVec2 size = { -1, -1 };
    /// (optional) preliminary window position
    const ImVec2* pos = 0;
    ImGuiWindowFlags flags = 0;
};
/// Same as ImGui::Begin, but with loading and saving the window position
/// \details see also \ref LoadSavedWindowPos and \ref SaveWindowPosition
MRVIEWER_API bool BeginSavedWindowPos( const std::string& name, bool* open, const SavedWindowPosParams& params );

/// begin state plugin window with custom style.  if you use this function, you must call EndCustomStatePlugin to close the plugin correctly.
/// the flags ImGuiWindowFlags_NoScrollbar and ImGuiWindow_NoScrollingWithMouse are forced in the function.
MRVIEWER_API bool BeginCustomStatePlugin( const char* label, bool* open, const CustomStatePluginWindowParameters& params = {} );
/// end state plugin window with custom style
MRVIEWER_API void EndCustomStatePlugin();

/// starts modal window with no animation for background
MRVIEWER_API bool BeginModalNoAnimation( const char* label, bool* open = nullptr, ImGuiWindowFlags flags = 0 );

/// Input int according valid from BitSet
/// \brief same as ImGui::InputInt
/// \return true if value was changed and valid
/// \note if inputted invalid value, it will be changed to first upper valid value (or find_last)\n
/// value can't be changed if bs has no any valid value
MRVIEWER_API bool InputIntBitSet( const char* label, int* v, const MR::BitSet& bs, int step = 1, int step_fast = 100, ImGuiInputTextFlags flags = 0 );

/**
 * \brief Combine of ImGui::DragInt and ImGui::InputInt
 * \details Value can be changed by drag or "+"/"-" buttons
 * \note used new visual style
 */
MRVIEWER_API bool DragInputInt( const char* label, int* value, float speed = 1, int min = std::numeric_limits<int>::lowest(),
                                int max = std::numeric_limits<int>::max(), const char* format = "%d", ImGuiSliderFlags flags = ImGuiSliderFlags_None );

/**
 * \brief Draw text as link, calls callback on click
 * \details Draw text as link, colored with blue, calls callback on click
 */
MRVIEWER_API bool Link( const char* label, uint32_t color = MR::Color( 60, 120, 255 ).getUInt32() );

/// return struct of ImGui::Palette \n
/// values are bits
enum class PaletteChanges
{
    None    = 0,
    Reset   = 1, // reset palette
    Texture = 2, // texture and legend must be updated
    Ranges  = 4, // uv-coordinates must be recomputed for the same values
    All = Texture | Ranges | Reset, // 0b111
};
MR_MAKE_FLAG_OPERATORS( PaletteChanges )

/// Helper palette widget, allows to change palette ranges and filter type \n
/// can load and save palette preset.
/// \param presetName stores the currently selected palette's preset name or empty string if the palette was edited by user
/// \param fixZero if present shows checkbox to fix zero symmetrical palette
/// \return mask of changes, if it has PaletteChanges::Texture bit - object requires texture update,
/// if it has PaletteChanges::Ranges uv coordinates should be recalculated and updated in object
MRVIEWER_API PaletteChanges Palette(
    const char* label,
    MR::Palette& palette,
    std::string& presetName,
    float width,
    float menuScaling,
    bool* fixZero = nullptr,
    float speed = 1.0f,
    float min = std::numeric_limits<float>::lowest(),
    float max = std::numeric_limits<float>::max()
);

// Parameters for the `Plane( MR::PlaneWidget& ... )` function
enum class PlaneWidgetFlags
{
    None = 0,   // Default setup
    DisableVisibility = 1   // Don't show "Show Plane" checkbox (and the preceding separator)
};
MR_MAKE_FLAG_OPERATORS( PlaneWidgetFlags )

/// Helper plane widget, allows to draw specified plain in the scene \n
/// can import plane from the scene, draw it with mouse or adjust with controls
/// planeWidget stores the plane widget params
MRVIEWER_API void Plane( MR::PlaneWidget& planeWidget, float menuScaling, PlaneWidgetFlags flags = {} );

/// Shows 3 edit boxes for editing of world direction coordinates;
/// \param editDragging must be initialized with zero, used to append only one history action on start dragging;
/// \param historyName the name of history action created on start dragging;
/// \return true if the direction was changed inside
MRVIEWER_API bool Direction( MR::DirectionWidget& dirWidget, bool& editDragging, const std::string& historyName );

/// draw image with Y-direction inversed up-down
MRVIEWER_API void Image( const MR::ImGuiImage& image, const ImVec2& size, const MR::Color& multColor );
MRVIEWER_API void Image( const MR::ImGuiImage& image, const ImVec2& size, const ImVec4& multColor = { 1, 1, 1, 1 } );

/// get image coordinates under cursor considering Y-direction flipping
MRVIEWER_API MR::Vector2i GetImagePointerCoord( const MR::ImGuiImage& image, const ImVec2& size, const ImVec2& imagePos );


/// draw spinner in given place, radius with respect to scaling
MRVIEWER_API void Spinner( float radius, float scaling );

/// draw big title with close cross (i.e. for settings modal popup )
MRVIEWER_API bool ModalBigTitle( const char* title, float scaling );

/// draw exit button with close cross (i.e. for settings modal popup )
MRVIEWER_API bool ModalExitButton( float scaling );

/// get exponential speed for this value
inline float getExpSpeed( float val, float frac = 0.01f, float min = 1e-5f )
    { return std::max( val * frac, min ); }

} // namespace ImGui
