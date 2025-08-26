#include "MRUIRectAllocator.h"

#include "MRViewer/MRViewer.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRImGuiVectorOperators.h"

#include <imgui_internal.h>

namespace MR::UI
{

RectAllocator::RectAllocator() {}

RectAllocator::FindFreeRectResult RectAllocator::findFreeRect(
    Box2f preferredRect,
    Box2f preferredBounds,
    FindPotentiallyOverlappingRects findOverlaps,
    ImVec2 axisWeights
)
{
    FindFreeRectResult bestRect{ .rect = preferredRect, .ok = false };

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

    visitedCoords.try_emplace( preferredRect.min, 0.0f );
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
        findOverlaps( thisRect, [&]( const char* name, Box2f otherRect )
        {
            (void)name;

            if ( !rectRectOverlap( thisRect, otherRect ) )
                return;

            anyIntersections = true;

            Box2f neighbors[4] = {
                { Vector2f( otherRect.max.x, thisRect.min.y ), Vector2f( otherRect.max.x + thisRect.size().x, thisRect.max.y ) }, // +X
                { Vector2f( thisRect.min.x, otherRect.max.y ), Vector2f( thisRect.max.x, otherRect.max.y + thisRect.size().y ) }, // +Y
                { Vector2f( otherRect.min.x - thisRect.size().x, thisRect.min.y ), Vector2f( otherRect.min.x, thisRect.max.y ) }, // -X
                { Vector2f( thisRect.min.x, otherRect.min.y - thisRect.size().y ), Vector2f( thisRect.max.x, otherRect.min.y ) }, // -Y
            };
            float deltaCosts[4] = {
                ( otherRect.max.x - thisRect.min.x ) * axisWeights[0],
                ( otherRect.max.y - thisRect.min.y ) * axisWeights[1],
                ( thisRect.max.x - otherRect.min.x ) * axisWeights[0],
                ( thisRect.max.y - otherRect.min.y ) * axisWeights[1],
            };

            for ( int i = 0; i < 4; i++ )
            {
                const Box2f& neighbor = neighbors[i];
                float cost = deltaCosts[i];
                assert( cost > 0 );
                cost += curCoords.cost;

                auto neighborOverlapWithBounds = getOverlapWithBounds( neighbor );
                bool badBoundsOverlap = false;
                for ( int j = 0; j < 4; j++ )
                {
                    if ( neighborOverlapWithBounds[j] > curCoords.overlapWithBounds[j] )
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
        } );

        if ( !anyIntersections )
        {
            // Use this rect.
            bestRect.rect = thisRect;
            bestRect.ok = true;
            break;
        }
    }

    return bestRect;
}

void WindowRectAllocator::setFreeNextWindowPos( const char* expectedWindowName, ImVec2 defaultPos, ImGuiCond cond, ImVec2 pivot )
{
    bool findLocation = false;
    ImGuiWindow* window = nullptr;

    if ( cond != 0 && cond != ImGuiCond_Always )
    {
        window = ImGui::FindWindowByName( expectedWindowName );
        if ( window )
        {
            auto [iter, isNew] = windows_.try_emplace( expectedWindowName );
            if ( isNew )
            {
                // if new window, request position calculation in next frame
                // to be sure that this frame action will not affect it (for example closing window that is present in this frame)
                iter->second.state_ = AllocationState::Requested;
            }
            else
            {
                if ( iter->second.state_ == AllocationState::Requested )
                {
                    findLocation = true;
                    defaultPos = window->Pos;
                }
                iter->second.state_ = AllocationState::Set; // validate that window is still present
            }
        }
    }

    if ( findLocation )
    {
        // push into application window if it is out
        const Vector2f appWindowSize = Vector2f( getViewerInstance().framebufferSize );
        auto windowBox = Box2f::fromMinAndSize( defaultPos, window->Size );
        defaultPos.x = std::clamp( windowBox.min.x, 0.0f, std::max( 0.0f, windowBox.min.x - ( windowBox.max.x - appWindowSize.x ) ) );
        defaultPos.y = std::clamp( windowBox.min.y, 0.0f, std::max( 0.0f, windowBox.min.y - ( windowBox.max.y - appWindowSize.y ) ) );
        windowBox = Box2f::fromMinAndSize( defaultPos, window->Size );


        Box2f viewportBounds = getViewerInstance().getViewportsBounds();
        Box2f boundsFixed = viewportBounds;
        boundsFixed.min.y = ImGui::GetIO().DisplaySize.y - boundsFixed.max.y;
        boundsFixed.max.y = boundsFixed.min.y + viewportBounds.size().y;

        auto result = findFreeRect( windowBox, boundsFixed, [&]( Box2f rect, std::function<void( const char*, Box2f )> func )
        {
            // Just output all the rects for now.
            // FIXME: An AABB tree would be nice here, for better performance.
            (void)rect;

            for ( const ImGuiWindow* win : ImGui::GetCurrentContext()->Windows )
            {
                if ( !win->WasActive && !win->Appearing )
                    continue; // Skip inactive windows.
                if ( win->Flags & ImGuiWindowFlags_ChildWindow )
                    continue; // Skip child windows.
                if ( win->Flags & ImGuiWindowFlags_Tooltip )
                    continue; // Skip tooltips.
                std::string_view winNameView = win->Name;
                if ( auto pos = winNameView.find( "##" ); pos != std::string_view::npos && winNameView.find( "[rect_allocator_ignore]", pos + 2 ) != std::string_view::npos)
                    continue; // Ignore if the name contains the special tag.
                if ( winNameView.starts_with( "##ToolTip_" ) )
                    continue; // Ignore ImGui tooltips.
                if ( std::strcmp( win->Name, expectedWindowName ) == 0 )
                    continue; // Skip the target window itself.
                func( win->Name, Box2f::fromMinAndSize( win->Pos, win->Size ) );
            }
        }, ImVec2( 5, 1 ) );

        defaultPos = ImGuiMath::round( ImVec2( result.rect.min ) + ImVec2( result.rect.size() ) * pivot );
        cond = ImGuiCond_Always;
    }

    ImGui::SetNextWindowPos( defaultPos, cond, pivot );
}

void WindowRectAllocator::invalidateClosedWindows()
{
    for ( auto it = windows_.begin(); it != windows_.end(); )
    {
        if ( it->second.state_ == AllocationState::None )
        {
            it = windows_.erase( it );
            continue;
        }
        else if ( it->second.state_ == AllocationState::Set )
            it->second.state_ = AllocationState::None; // we will validate that this window is still present when window Begin function will be called

        ++it;
    }
}

WindowRectAllocator& getDefaultWindowRectAllocator()
{
    static WindowRectAllocator ret;
    return ret;
}

ImVec2 LabelRectAllocator::createRect( ViewportId viewportId, std::string id, ImVec2 pos, ImVec2 size, bool forceExactPosition )
{
    // FIXME: An AABB tree would be nice here, for better performance.

    if ( lastFrameCount_ != ImGui::GetFrameCount() )
    {
        lastFrameCount_ = ImGui::GetFrameCount();

        for ( auto& perViewportMap : entries_ )
        {
            for ( auto it = perViewportMap.begin(); it != perViewportMap.end(); )
            {
                if ( !std::exchange( it->second.visitedThisFrame, false ) )
                {
                    it = perViewportMap.erase( it );
                    continue;
                }

                ++it;
            }
        }
    }

    Viewer& viewer = getViewerInstance();
    std::size_t viewportIndex = viewer.viewport_index( viewportId );

    if ( viewportIndex >= entries_.size() )
        entries_.resize( viewportIndex + 1 );
    auto& perViewportMap = entries_[viewportIndex];

    if ( forceExactPosition )
    {
        auto [iter, isNew] = perViewportMap.try_emplace( std::move( id ) );
        iter->second.visitedThisFrame = true;
        iter->second.box = Box2f::fromMinAndSize( pos, size );
    }

    Viewport& viewport = viewer.viewport( viewportId );
    Box2f viewportRect = viewport.getViewportRect();

    FindFreeRectResult result = findFreeRect(
        Box2f::fromMinAndSize( pos, size ),
        Box2f( ImVec2( viewportRect.min.x, viewer.framebufferSize.y - viewportRect.max.y ), ImVec2( viewportRect.max.x, viewer.framebufferSize.y - viewportRect.min.y ) ),
        [&]( Box2f target, std::function<void( const char* name, Box2f box )> overlaps )
        {
            // Ugh, linear search.
            for ( const auto& elem : perViewportMap )
            {
                if ( elem.first == id )
                    continue; // No self-overlap.
                if ( rectRectOverlap( target, elem.second.box ) )
                    overlaps( elem.first.c_str(), elem.second.box );
            }
        },
        ImVec2( 1, 1 )
    );

    auto [iter, isNew] = perViewportMap.try_emplace( std::move( id ) );
    iter->second.visitedThisFrame = true;
    iter->second.box = result.rect;

    return result.rect.min;
}

LabelRectAllocator& getDefaultLabelRectAllocator()
{
    static LabelRectAllocator ret;
    return ret;
}

}
