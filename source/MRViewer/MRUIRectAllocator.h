#pragma once

#include "exports.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRHash.h"
#include "MRMesh/MRVector2.h"

#include <imgui.h>
#include <parallel_hashmap/phmap.h>

namespace MR::UI
{

// A generic rect allocator.
class RectAllocator
{
public:
    MRVIEWER_API RectAllocator();

    struct FindFreeRectResult
    {
        Box2f rect;
        bool ok = true; // False if no free space for this rect.
        [[nodiscard]] explicit operator bool() const { return ok; }
    };

    // Given an input rect, this function must find all POTENTIALLY overlapping rects ("overlapping" means according to `rectRectOverlap()`).
    using FindPotentiallyOverlappingRects = std::function<void( Box2f target, std::function<void( const char* name, Box2f box )> overlaps )>;

    // Finds a free rectangle of the specified size, as close as possible to the specified position.
    // On failure, returns `.ok == false` and returns the input rect unchanged.
    [[nodiscard]] MRVIEWER_API FindFreeRectResult findFreeRect(
        Box2f preferredRect,
        // The outer bounds that we try to fit the rect into.
        Box2f preferredBounds,
        // Given any rect, this must return all existing rects potentially overlapping with it.
        FindPotentiallyOverlappingRects findOverlaps,
        // This sets the preference for X and Y axes. Larger number = less likely to shift over that axis.
        ImVec2 axisWeights = ImVec2( 1, 1 )
    );

    // Checks if two rects overlap.
    // We can't use `.intersects()` here because we want `.max` to be exclusive, while `.intersects()` treats both bounds as inclusive.
    [[nodiscard]] static bool rectRectOverlap( Box2f a, Box2f b )
    {
        return a.max.x > b.min.x && a.min.x < b.max.x && a.max.y > b.min.y && a.min.y < b.max.y;
    }

private:
    // Maps coords to cost.
    phmap::flat_hash_map<Vector2f, float> visitedCoords;

    struct CoordsToVisit
    {
        Vector2f pos;
        float cost = 0;
        std::array<float, 4> overlapWithBounds;
    };
    std::vector<CoordsToVisit> coordsToVisitHeap;
};

// A rect allocator specifically for ImGui windows.
class WindowRectAllocator : public RectAllocator
{
public:
    // Call this before drawing a window.
    // `expectedWindowName` must match the window name.
    // The remaining parameters are forwarded to `ImGui::SetNextWindowPos()`, expect for one time where we find a free rect and use it instead.
    // `cond` must not be `ImGuiCond_Always` (aka 0), in that case we just forward the arguments and don't try to find a rect.
    MRVIEWER_API void setNextWindowPos( const char* expectedWindowName, ImVec2 defaultPos, ImGuiCond cond = ImGuiCond_Appearing, ImVec2 pivot = ImVec2() );

    // Must be called once every frame.
    MRVIEWER_API void preTick();

private:
    struct WindowEntry
    {
        bool visitedThisFrame = true;
    };
    phmap::flat_hash_map<std::string, WindowEntry> windows;
};
[[nodiscard]] MRVIEWER_API WindowRectAllocator& getDefaultWindowRectAllocator();

}
