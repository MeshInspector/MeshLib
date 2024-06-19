#include "MRUIRectAllocator.h"

#include <imgui.h>

namespace MR::UI
{

RectAllocator::RectAllocator() {}

RectAllocator::MakeRectResult RectAllocator::makeRect(
    std::string_view name,
    std::function<Box2f()> getPreferredLocation,
    MakeRectFlags flags,
    std::size_t forceExactLocationIfOlderThan
)
{
    // This will be true if the `forceUpdate` flag is set, even if the rect isn't actually new.
    bool isNew = false;
    // And this doesn't respect the flag.
    bool isActuallyNew = false;

    std::size_t age = 0;

    // Only makes sense if `isNew == false`.
    decltype(currentRects)::iterator iter = currentRects.find( name );
    if ( iter != currentRects.end() )
        age = iter->second.age;
    else
        isActuallyNew = true;

    if ( forceExactLocationIfOlderThan && age >= forceExactLocationIfOlderThan )
        flags |= MakeRectFlags::forceExactLocation;

    if ( bool( flags & MakeRectFlags::forceUpdate ) )
    {
        if ( bool( flags & MakeRectFlags::forceExactLocation ) )
        {
            if ( isActuallyNew )
                iter = currentRects.try_emplace( name ).first;

            iter->second.seenThisFrame = true;
            iter->second.age = age;
            return { .rect = iter->second.rect = getPreferredLocation(), .ok = true, .age = age };
        }

        if ( !isActuallyNew )
            currentRects.erase( iter );
        iter = {};
        isNew = true;
    }
    else
    {
        isNew = isActuallyNew;
    }

    if ( !isNew )
    {
        iter->second.seenThisFrame = true;
        return { .rect = iter->second.rect, .ok = true, .age = age };
    }

    if ( bool( flags & MakeRectFlags::forceExactLocation ) )
    {
        auto result = currentRects.try_emplace( name );
        assert( result.second && "Why does the rect already exist?" );
        result.first->second.seenThisFrame = true;
        result.first->second.age = age;
        return { .rect = result.first->second.rect = getPreferredLocation(), .ok = true, .age = age };
    }

    // Find a free rect.

    const Box2f preferredRect = getPreferredLocation();
    MakeRectResult bestRect{ .rect = preferredRect, .ok = false, .age = age };

    visitedCoords.clear();
    coordsToVisitHeap.clear();

    auto getOverlapWithBounds = [&]( Box2f rect ) -> std::array<float, 4>
    {
        std::array<float, 4> ret = {
            rect.max.x - preferredBounds.max.x,
            rect.max.y - preferredBounds.max.y,
            preferredBounds.min.x - rect.min.x,
            preferredBounds.min.y - rect.min.y,
        };

        for ( float& x : ret )
        {
            if ( x < 0 )
                x = 0;
        }
        return ret;
    };

    visitedCoords.try_emplace( preferredRect.min, 0 );
    coordsToVisitHeap.push_back( { .pos = preferredRect.min, .cost = 0, .overlapWithBounds = getOverlapWithBounds( preferredRect ) } );

    auto heapComparator = []( const CoordsToVisit& a, const CoordsToVisit& b )
    {
        // Min (sic!) heap by cost.
        return a.cost > b.cost;
    };

    while ( !coordsToVisitHeap.empty() )
    {
        const CoordsToVisit curCoords = coordsToVisitHeap.front();
        std::pop_heap( coordsToVisitHeap.begin(), coordsToVisitHeap.end(), heapComparator );
        coordsToVisitHeap.pop_back();

        const Box2f thisRect = Box2f::fromMinAndSize( curCoords.pos, preferredRect.size() );

        // Intersect against existing rects.
        bool anyIntersections = false;
        for ( const auto& [otherName, otherRectData] : currentRects )
        {
            const Box2f otherRect = otherRectData.rect;

            // Not using `.intersects()` here because we want `.max` to be exclusive, while `.intersects()` treats both bounds as inclusive.
            if ( thisRect.max.x <= otherRect.min.x || thisRect.min.x >= otherRect.max.x || thisRect.max.y <= otherRect.min.y || thisRect.min.y >= otherRect.max.y )
                continue;

            anyIntersections = true;

            Box2f neighbors[4] = {
                { Vector2f( otherRect.max.x, thisRect.min.y ), Vector2f( otherRect.max.x + thisRect.size().x, thisRect.max.y ) }, // +X
                { Vector2f( thisRect.min.x, otherRect.max.y ), Vector2f( thisRect.max.x, otherRect.max.y + thisRect.size().y ) }, // +Y
                { Vector2f( otherRect.min.x - thisRect.size().x, thisRect.min.y ), Vector2f( otherRect.min.x, thisRect.max.y ) }, // -X
                { Vector2f( thisRect.min.x, otherRect.min.y - thisRect.size().y ), Vector2f( thisRect.max.x, otherRect.min.y ) }, // -Y
            };
            float deltaCosts[4] = {
                otherRect.max.x - thisRect.min.x,
                otherRect.max.y - thisRect.min.y,
                thisRect.max.x - otherRect.min.x,
                thisRect.max.y - otherRect.min.y,
            };

            for ( int i = 0; i < 4; i++ )
            {
                const Box2f& neighbor = neighbors[i];
                float cost = deltaCosts[i];
                assert( cost > 0 );
                cost += curCoords.cost;

                auto neighborOverlapWithBounds = getOverlapWithBounds( neighbor );
                bool badBoundsOverlap = false;
                for ( int i = 0; i < 4; i++ )
                {
                    if ( neighborOverlapWithBounds[i] > curCoords.overlapWithBounds[i] )
                    {
                        badBoundsOverlap = true;
                        break;
                    }
                }
                if ( badBoundsOverlap )
                    continue; // This rect is more out of bounds than the previous one, ignore.

                auto [visIter, visNew] = visitedCoords.try_emplace( neighbor.min, cost );
                if ( !visNew )
                {
                    if ( visIter->second <= cost )
                        continue; // Already visited with less cost, ignore.
                    visIter->second = cost;
                }

                coordsToVisitHeap.push_back( { .pos = neighbor.min, .cost = cost, .overlapWithBounds = neighborOverlapWithBounds } );
                std::push_heap( coordsToVisitHeap.begin(), coordsToVisitHeap.end(), heapComparator );
            }
        }

        if ( !anyIntersections )
        {
            // Use this rect.
            bestRect.rect = thisRect;
            bestRect.ok = true;
            break;
        }
    }

    currentRects.try_emplace( name, CurRect{ .rect = bestRect.rect, .seenThisFrame = true, .age = age } );

    return bestRect;
}

void RectAllocator::preTick( Box2f bounds )
{
    preferredBounds = bounds;

    for ( auto it = currentRects.begin(); it != currentRects.end(); )
    {
        if ( !std::exchange( it->second.seenThisFrame, false ) )
        {
            it = currentRects.erase( it );
            continue;
        }
        it->second.age++;
        ++it;
    }
}

void RectAllocator::debugVisualize()
{
    ImDrawList& list = *ImGui::GetForegroundDrawList();

    list.AddRect( preferredBounds.min, preferredBounds.max, ImColor( 1.f, 1.f, 0.f, 1.f ), 8 );

    for ( const auto& [ name, data ] : currentRects )
    {
        list.AddRect( data.rect.min, data.rect.max, ImColor( 1.f, 0.f, 0.f, 1.f ), 8 );
        list.AddText( data.rect.min + Vector2f( 5, 5 ), ImColor( 1.f, 0.f, 0.f, 1.f ), name.c_str() );
    }
}

RectAllocator& getDefaultRectAllocator()
{
    static RectAllocator ret;
    return ret;
}

}
