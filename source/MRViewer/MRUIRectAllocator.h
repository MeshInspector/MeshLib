#pragma once

#include "exports.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRFlagOperators.h"
#include "MRMesh/MRHash.h"
#include "MRMesh/MRVector2.h"

#include <parallel_hashmap/phmap.h>

namespace MR::UI
{

class RectAllocator
{
public:
    MRVIEWER_API RectAllocator();

    enum class MakeRectFlags
    {
        // Call `getPreferredLocation()` and update the rect location, even if the rect already existed on the previous frame.
        forceUpdate = 1 << 0,
        // Don't try to find free space, use the exact rect specified (will return the input rect unchanged).
        forceExactLocation = 1 << 1,
    };
    MR_MAKE_FLAG_OPERATORS_IN_CLASS(MakeRectFlags)

    struct MakeRectResult
    {
        Box2f rect;
        bool ok = true; // False if no free space for this rect.
        std::size_t age = 0; // This is incremented every frame while this rect exists. Initially 0.
        [[nodiscard]] explicit operator bool() const { return ok; }
    };

    // Call this every frame to maintain a rectangle.
    // `name` is a unique name. If you reuse it in the same frame, the existing rect will be overridden.
    // `getPreferredLocation()` will be called once when adding the rect, to get its size and preferred position.
    // The initial position will be choosen as a free position nearest to the preferred once, and then will stay unchanged.
    // Returns the current rect position. If there's no free rect, returns `.ok = false` and the input rectangle.
    [[nodiscard]] MRVIEWER_API MakeRectResult makeRect(
        std::string_view name,
        std::function<Box2f()> getPreferredLocation,
        MakeRectFlags flags = {},
        // If this is not zero and the rect has existed for at least this many frames, implicitly adds flag `forceExactLocation`.
        std::size_t forceExactLocationIfOlderThan = 0
    );

    // Call this every frame, preferably before adding all the rects.
    // `bounds` are the preferred bounds into which to fit the rects.
    MRVIEWER_API void preTick( Box2f bounds );

    // Visualize the rects for debugging, using `ImGui`.
    MRVIEWER_API void debugVisualize();

private:
    Box2f preferredBounds;

    struct CurRect
    {
        Box2f rect;
        bool seenThisFrame = true;
        std::size_t age = 0;
    };

    phmap::flat_hash_map<std::string, CurRect> currentRects;

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

[[nodiscard]] MRVIEWER_API RectAllocator& getDefaultRectAllocator();

}
