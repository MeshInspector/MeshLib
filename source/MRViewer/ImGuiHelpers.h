#pragma once
// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2018 Jérémie Dumas <jeremie.dumas@ens-lyon.org>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "MRMesh/MRMeshFwd.h"
#include "exports.h"
#include "ImGuiTraits.h"
#include "MRMesh/MRColor.h"
#include <algorithm>
#include <functional>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace MR
{
class Palette;
class ImGuiImage;
}

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
  char buf[1024];
  std::fill_n(buf, 1024, char(0));
  std::copy_n(str.begin(), std::min(1024, (int) str.size()), buf);
  if (ImGui::InputText(label, buf, 1024, flags, callback, user_data))
  {
    str = std::string(buf);
    return true;
  }
  return false;
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

MRVIEWER_API void PlotCustomHistogram( const char* str_id,
                                 std::function<float( int idx )> values_getter,
                                 std::function<void( int idx )> tooltip,
                                 std::function<void( int idx )> on_click,
                                 int values_count, int values_offset = 0,
                                 float scale_min = FLT_MAX, float scale_max = FLT_MAX,
                                 ImVec2 frame_size = ImVec2( 0, 0 ), int selectedBarId = -1 );

/// begin typical state plugin window
MRVIEWER_API bool BeginStatePlugin( const char* label, bool* open, float width );

/// begin state plugin window with custom style.  if you use this function, you must call EndCustomStatePlugin to close the plugin correctly.
/// the flags ImGuiWindowFlags_NoScrollbar and ImGuiWindow_NoScrollingWithMouse are forced in the function.
/// if the plugin supports resizing, you must pass changedSize argument where are stored resized width and height
MRVIEWER_API bool BeginCustomStatePlugin( const char* label, bool* open, bool* collapsed, float width, float menuScaling, float height = 0.0f, 
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize, ImVec2* changedSize = nullptr );
/// end state plugin window with custom style
MRVIEWER_API void EndCustomStatePlugin();

/// starts modal window with no animation for background
MRVIEWER_API bool BeginModalNoAnimation( const char* label, bool* open = nullptr, ImGuiWindowFlags flags = 0 );

/// draw a button, which can be disabled (valid = false)
MRVIEWER_API bool ButtonValid( const char* label, bool valid, const ImVec2& size = ImVec2(0, 0) );

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
    None,                  // 0b00
    Texture,               // 0b01
    Ranges,                // 0b10
    All = Texture | Ranges // 0b11
};

/// Helper palette widget, allows to change palette ranges and filter type \n
/// can load and save palette preset.
/// \param fixZero if present shows checkbox to fix zero symmetrical palette
/// \return mask of changes, if it has PaletteChanges::Texture bit - object requires texture update,
/// if it has PaletteChanges::Ranges uv coordinates should be recalculated and updated in object
MRVIEWER_API PaletteChanges Palette( 
    const char* label,
    MR::Palette& palette,
    float width,
    float menuScaling,
    bool* fixZero = nullptr,
    float speed = 1.0f,
    float min = std::numeric_limits<float>::lowest(),
    float max = std::numeric_limits<float>::max(),
    const char* format = "%.3f" );


/// draw image with Y-direction inversed up-down
MRVIEWER_API void Image( const MR::ImGuiImage& image, const ImVec2& size, const MR::Color& multColor );
MRVIEWER_API void Image( const MR::ImGuiImage& image, const ImVec2& size, const ImVec4& multColor = { 1, 1, 1, 1 } );

/// get image coordinates under cursor considering Y-direction flipping
MRVIEWER_API MR::Vector2i GetImagePointerCoord( const MR::ImGuiImage& image, const ImVec2& size, const ImVec2& imagePos );

/// draw tooltip only if current item is hovered
MRVIEWER_API void SetTooltipIfHovered( const std::string& text, float scaling );
///add text with separator line 
MRVIEWER_API void Separator( float scaling, const std::string& text = "" );

} // namespace ImGui
