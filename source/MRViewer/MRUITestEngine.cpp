#include "MRUITestEngine.h"
#include "MRImGui.h"
#include "MRPch/MRFmt.h"

#include <imgui_internal.h>

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
    // The root element group.
    GroupEntry root;

    // The frame counter, as per `ImGui::GetFrameCount()`. When this changes, we prune dead elements.
    int curFrame = -1;

    // The stack for `pushTree()`, `popTree()`. Always has at least one element.
    std::vector<GroupEntry*> stack = { &root };

    // True if a TE-driven action (simulated click or value override) ran this frame.
    // Cleared at frame boundary in `checkForNewFrame()`.
    bool frameTriggered = false;

    // Paths staged for the next TE-triggered file dialog to return. Single-shot.
    std::vector<std::filesystem::path> stagedFileDialogPaths;

    // Status messages emitted during TE-driven actions; drained by MCP after each dispatch.
    std::vector<std::string> statusMessages;
};
// Our global state. Stores the current tree of buttons and button groups.
State state;

// True if the widget currently being submitted is inside an `ImGui::BeginDisabled` scope.
bool imGuiContextSaysDisabled()
{
    ImGuiContext* g = ImGui::GetCurrentContext();
    return g && ( g->CurrentItemFlags & ImGuiItemFlags_Disabled ) != 0;
}

// If a blocking modal popup is currently open and the widget being submitted is *outside* that
// modal's window tree (walking ImGui's ParentWindow chain), returns a view of the modal's window
// name. Otherwise returns empty (no modal open, or the widget is inside the modal).
// The returned view is valid only while ImGui state is untouched (i.e. during the same callback).
std::string_view imGuiBlockingModalName()
{
    ImGuiWindow* topModal = ImGui::GetTopMostPopupModal();
    if ( !topModal )
        return {};
    for ( ImGuiWindow* w = ImGui::GetCurrentWindow(); w; w = w->ParentWindow )
        if ( w == topModal )
            return {};
    return topModal->Name ? std::string_view{ topModal->Name } : std::string_view{ "<unnamed>" };
}

// Produce the effective disabled-reason string to store on an entry, given caller-supplied attrs.
// - If caller passed a reason, use it verbatim (takes precedence).
// - Else if ImGui says the widget is drawn under BeginDisabled, use a generic fallback so the
//   entry is still marked disabled even though the caller didn't know why.
// - Else, if a blocking modal popup is open and the widget is drawn outside it, return
//   "blocked by modal '<name>'" — the widget can't receive input while the modal is on top.
// - Else empty (entry accepts input).
std::string effectiveDisabledReason( const EntryAttributes& attrs )
{
    if ( !attrs.disabledReason.empty() )
        return std::string( attrs.disabledReason );
    if ( imGuiContextSaysDisabled() )
        return "drawn inside ImGui::BeginDisabled";
    if ( const auto modal = imGuiBlockingModalName(); !modal.empty() )
        return fmt::format( "blocked by modal '{}'", modal );
    return {};
}

void checkForNewFrame()
{
    // If this is a new frame...
    if ( ImGui::GetFrameCount() != state.curFrame )
    {
        state.curFrame = ImGui::GetFrameCount();
        state.frameTriggered = false;

        // Make sure the tree stack is fine.
        assert(state.stack.size() == 1 && "Missing `UI::TestEngine::popTree()`.");

        // Prune stale elements from the tree.
        auto pruneDeadEntries = [&]( auto& pruneDeadEntries, GroupEntry& group ) -> void
        {
            for ( auto it = group.elems.begin(); it != group.elems.end(); )
            {
                if ( !it->second.visitedOnThisFrame )
                {
                    it = group.elems.erase( it );
                }
                else
                {
                    it->second.visitedOnThisFrame = false;

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

template <typename T>
std::optional<T> detail::createValueLow( std::string_view name, std::optional<BoundedValue<T>> value, bool consumeValueOverride /*= true*/, const EntryAttributes& attrs /*= {}*/ )
{
    #if MR_ENABLE_UI_TEST_ENGINE

    checkForNewFrame();

    auto& map = state.stack.back()->elems;
    auto iter = map.find( name ); // I wish I could use `std::try_emplace` here directly...
    if ( iter == map.end() )
    {
        iter = map.try_emplace( std::string( name ) ).first;
    }
    else
    {
        // If you see this assert, you likely have duplicate drag/slider names.
        assert( !iter->second.visitedOnThisFrame && "Registering the same entry more than once in a single frame!" );
    }

    ValueEntry* entry = std::get_if<ValueEntry>( &iter->second.value );
    if ( !entry )
        entry = &iter->second.value.emplace<ValueEntry>();

    entry->disabledReason = effectiveDisabledReason( attrs );

    std::optional<T> ret;

    ValueEntry::Value<T>* val = std::get_if<ValueEntry::Value<T>>( &entry->value );
    if ( val )
    {
        if ( consumeValueOverride )
        {
            ret = std::move( val->simulatedValue );
            val->simulatedValue = {};
        }
        else
        {
            ret = val->simulatedValue;
        }
        if ( ret )
            state.frameTriggered = true;
    }
    else
    {
        val = &entry->value.emplace<ValueEntry::Value<T>>();
    }

    if ( value )
    {
        iter->second.visitedOnThisFrame = true;
        val->value = std::move( value->value ); // Could also read `ret` here, but that would be a bit weird, I guess?
        if constexpr ( std::is_same_v<T, std::string> )
        {
            val->allowedValues = value->allowedValues;
        }
        else
        {
            val->min = value->min;
            val->max = value->max;
        }
    }

    return ret;

    #else
    return {};
    #endif
}

template std::optional<std::int64_t> detail::createValueLow( std::string_view name, std::optional<BoundedValue<std::int64_t>> value, bool consumeValueOverride, const EntryAttributes& attrs );
template std::optional<std::uint64_t> detail::createValueLow( std::string_view name, std::optional<BoundedValue<std::uint64_t>> value, bool consumeValueOverride, const EntryAttributes& attrs );
template std::optional<double> detail::createValueLow( std::string_view name, std::optional<BoundedValue<double>> value, bool consumeValueOverride, const EntryAttributes& attrs );
template std::optional<std::string> detail::createValueLow( std::string_view name, std::optional<BoundedValue<std::string>> value, bool consumeValueOverride, const EntryAttributes& attrs );

bool createButton( std::string_view name, const EntryAttributes& attrs )
{
    #if MR_ENABLE_UI_TEST_ENGINE
    checkForNewFrame();

    auto& map = state.stack.back()->elems;
    auto iter = map.find( name ); // I wish I could use `std::try_emplace` here directly...
    if ( iter == map.end() )
    {
        iter = map.try_emplace( std::string( name ) ).first;
    }
    else
    {
        // If you see this assert, you likely have duplicate button names.
        // If you truly have several buttons with the same name in one place, add unique `##...` suffixes to them.
        // If the buttons are in different parts of the application and shouldn't collide,
        // use `UI::TestEngine::pushTree("...")` and `popTree()` to group them into named groups, with unique names in each group.
        assert( !iter->second.visitedOnThisFrame && "Registering the same entry more than once in a single frame!" );
    }

    ButtonEntry* button = std::get_if<ButtonEntry>( &iter->second.value );
    if ( !button )
        button = &iter->second.value.emplace<ButtonEntry>();

    button->disabledReason = effectiveDisabledReason( attrs );

    iter->second.visitedOnThisFrame = true;

    // commented, because it is already logged in MRPythonUiInteraction.cpp/pressButton
    // if ( button->simulateClick )
    //    spdlog::info( "Button {} click simulation", name );

    const bool clicked = std::exchange( button->simulateClick, false );
    if ( clicked )
        state.frameTriggered = true;
    return clicked;
    #else
    (void)attrs;
    return false;
    #endif
}

std::optional<std::string> createValue( std::string_view name, std::string value, bool consumeValueOverride, std::optional<std::vector<std::string>> allowedValues, const EntryAttributes& attrs )
{
    return detail::createValueLow<std::string>( name, detail::BoundedValue<std::string>{ .value = std::move( value ), .allowedValues = std::move( allowedValues ) }, consumeValueOverride, attrs );
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
        assert( !iter->second.visitedOnThisFrame && "Registering the same entry more than once in a single frame!" );

    GroupEntry* subgroup = std::get_if<GroupEntry>( &iter->second.value );
    if ( !subgroup )
        subgroup = &iter->second.value.emplace<GroupEntry>();

    iter->second.visitedOnThisFrame = true;

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

std::string_view Entry::getKindName() const
{
    return std::visit( []<typename T>( const T & ){ return T::kindName; }, value );
}

const GroupEntry& getRootEntry()
{
    return state.root;
}

bool wasFrameTriggered()
{
    #if MR_ENABLE_UI_TEST_ENGINE
    return state.frameTriggered;
    #else
    return false;
    #endif
}

void stageFileDialogPaths( std::vector<std::filesystem::path> paths )
{
    #if MR_ENABLE_UI_TEST_ENGINE
    state.stagedFileDialogPaths = std::move( paths );
    #else
    (void)paths;
    #endif
}

std::vector<std::filesystem::path> consumeStagedFileDialogPaths()
{
    #if MR_ENABLE_UI_TEST_ENGINE
    return std::exchange( state.stagedFileDialogPaths, {} );
    #else
    return {};
    #endif
}

void appendStatusMessage( std::string msg )
{
    #if MR_ENABLE_UI_TEST_ENGINE
    state.statusMessages.push_back( std::move( msg ) );
    #else
    (void)msg;
    #endif
}

std::vector<std::string> consumeStatusMessages()
{
    #if MR_ENABLE_UI_TEST_ENGINE
    return std::exchange( state.statusMessages, {} );
    #else
    return {};
    #endif
}

[[nodiscard]] MRVIEWER_API Unexpected<std::string> Entry::unexpected_( std::string_view selfName, std::string_view tKindName )
{
    if ( selfName.empty() )
        return unexpected( fmt::format( "Expected UI entity to be a `{}` but got a `{}`.", tKindName, getKindName() ) );
    else
        return unexpected( fmt::format( "Expected UI entity `{}` to be a `{}` but got a `{}`.", selfName, tKindName, getKindName() ) );
}

} // namespace MR::UI::TestEngine
