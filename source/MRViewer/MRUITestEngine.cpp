#include "MRUITestEngine.h"

#include <imgui.h>

#ifndef MR_ENABLE_UI_TEST_ENGINE
// Set to 0 to disable the UI test engine. All functions will act as if no UI elements are registered.
#define MR_ENABLE_UI_TEST_ENGINE 1
#endif

namespace MR::UI::TestEngine
{

namespace
{

struct State
{
    GroupEntry root;
    int curFrame = -1;

    std::vector<GroupEntry*> stack = { &root };
};
State state;

void checkForNewFrame()
{
    // If this is a new frame...
    if ( ImGui::GetFrameCount() != state.curFrame )
    {
        state.curFrame = ImGui::GetFrameCount();

        // Make sure the tree stack is fine.
        assert(state.stack.size() == 1 && "Missing `UI::TestEngine::popTree()`.");

        // Prune stale elements from the tree.
        auto pruneDeadEntries = [&]( auto& pruneDeadEntries, GroupEntry& group ) -> void
        {
            for ( auto it = group.elems.begin(); it != group.elems.end(); )
            {
                if ( !it->second.alive )
                {
                    it = group.elems.erase( it );
                }
                else
                {
                    it->second.alive = false;

                    if ( auto subgroup = std::get_if<GroupEntry>( &it->second.value ) )
                        pruneDeadEntries( pruneDeadEntries, *subgroup );

                    ++it;
                    continue;
                }
            }
        };
        pruneDeadEntries( pruneDeadEntries, state.root );
    }
}

} // namespace

bool createButton( std::string_view name )
{
    #if MR_ENABLE_UI_TEST_ENGINE
    checkForNewFrame();

    auto& map = state.stack.back()->elems;
    auto iter = map.find( name ); // I wish I could use `std::try_emplace` here directly...
    if ( iter == map.end() )
        iter = map.try_emplace( std::string( name ) ).first;
    else
        assert( !iter->second.alive && "Registering the same entry more than once in a single frame!" );

    ButtonEntry* button = std::get_if<ButtonEntry>( &iter->second.value );
    if ( !button )
        button = &iter->second.value.emplace<ButtonEntry>();

    iter->second.alive = true;

    return std::exchange( button->simulateClick, false );
    #endif
}

void pushTree( std::string_view name )
{
    #if MR_ENABLE_UI_TEST_ENGINE
    checkForNewFrame();

    auto& map = state.stack.back()->elems;
    auto iter = map.find( name ); // I wish I could use `std::try_emplace` here directly...
    if ( iter == map.end() )
        iter = map.try_emplace( std::string( name ) ).first;
    else
        assert( !iter->second.alive && "Registering the same entry more than once in a single frame!" );

    GroupEntry* subgroup = std::get_if<GroupEntry>( &iter->second.value );
    if ( !subgroup )
        subgroup = &iter->second.value.emplace<GroupEntry>();

    iter->second.alive = true;

    state.stack.push_back( subgroup );
    #endif
}

void popTree()
{
    #if MR_ENABLE_UI_TEST_ENGINE
    checkForNewFrame();

    assert( state.stack.size() > 1 && "Excessive `UI::TestEngine::popTree()`." );
    if ( state.stack.size() <= 1 )
        return;

    state.stack.pop_back();
    #endif
}

const GroupEntry& getRootEntry()
{
    return state.root;
}

}
