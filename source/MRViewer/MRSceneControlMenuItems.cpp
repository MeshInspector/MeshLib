#include "MRSceneControlMenuItems.h"
#include "MRViewer.h"
#include "MRMakeSlot.h"
#include "MRHistoryStore.h"
#include "MRAppendHistory.h"
#include "MRSwapRootAction.h"
#include "MRRibbonMenu.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRCommandLoop.h"
#include "MRProgressBar.h"
#include "ImGuiHelpers.h"
#include "MRLambdaRibbonItem.h"
#include "MRPch/MRSpdlog.h"
#include <array>

namespace
{
constexpr unsigned cMaxNumActionsInList = 30u;
}

namespace MR
{

// Remove `##` and everything after it from the string.
// If the string doesn't contain it, return sit unchanged.
static std::string trimHashHashSuffix( std::string str )
{
    if ( auto sep = str.find( "##" ); sep != std::string::npos )
        str.resize( sep );
    return str;
}

UndoMenuItem::UndoMenuItem() :
    RibbonMenuItem( "Undo" )
{
    type_ = RibbonItemType::ButtonWithDrop;
    // deferred to be sure that viewer history is initialized
    CommandLoop::appendCommand( [&]
    {
        auto history = getViewerInstance().getGlobalHistoryStore();
        if ( !history )
            return;
        if ( !historyStoreConnection_.connected() )
        {
            historyStoreConnection_ = history->changedSignal.connect( MAKE_SLOT( &UndoMenuItem::updateUndoListCache_ ) );
            updateUndoListCache_( *history, HistoryStore::ChangeType::AppendAction ); // can by any type
        }
    } );
}

bool UndoMenuItem::action()
{
    Viewer::instanceRef().globalHistoryUndo();
    return false;
}

std::string UndoMenuItem::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    const auto& history = Viewer::instanceRef().getGlobalHistoryStore();
    if ( !history )
        return "Internal history stack is unavailable.";
    if ( dropList_.empty() )
        return "Nothing to undo.";
    return "";
}

std::string UndoMenuItem::getDynamicTooltip() const
{
    std::string res;
    if ( const auto& history = Viewer::instanceRef().getGlobalHistoryStore() )
        res = trimHashHashSuffix( history->getLastActionName( HistoryAction::Type::Undo ) );
    return res;
}

void UndoMenuItem::updateUndoListCache_( const HistoryStore& store, HistoryStore::ChangeType )
{
    auto lastUndos = store.getNActions( cMaxNumActionsInList, HistoryAction::Type::Undo );
    if ( lastUndos.empty() )
    {
        dropList_.clear();
        return;
    }

    dropList_.resize( lastUndos.size() );
    for ( int i = 0; i < lastUndos.size(); ++i )
    {
        dropList_[i] = std::make_shared<LambdaRibbonItem>( lastUndos[i] + "##" + std::to_string( i ),
            [history = Viewer::instanceRef().getGlobalHistoryStore(), i] ()
        {
            if ( !history )
                return;
            for ( int j = 0; j <= i; ++j )
                history->undo();
        } );
    }
}

RedoMenuItem::RedoMenuItem() :
    RibbonMenuItem( "Redo" )
{
    type_ = RibbonItemType::ButtonWithDrop;
    // deferred to be sure that viewer history is initialized
    CommandLoop::appendCommand( [&]
    {
        if ( !HistoryStore::getViewerInstance() )
            return;
        if ( !historyStoreConnection_.connected() )
        {
            historyStoreConnection_ = HistoryStore::getViewerInstance()->changedSignal.connect( MAKE_SLOT( &RedoMenuItem::updateRedoListCache_ ) );
            updateRedoListCache_( *HistoryStore::getViewerInstance(), HistoryStore::ChangeType::AppendAction ); // can by any type
        }
    } );
}

bool RedoMenuItem::action()
{
    Viewer::instanceRef().globalHistoryRedo();
    return false;
}

std::string RedoMenuItem::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    if ( !HistoryStore::getViewerInstance() )
        return "Internal history stack is unavailable.";
    if ( dropList_.empty() )
        return "Nothing to redo.";
    return "";
}

std::string RedoMenuItem::getDynamicTooltip() const
{
    std::string res;
    if ( auto history = HistoryStore::getViewerInstance() )
        res = trimHashHashSuffix( history->getLastActionName( HistoryAction::Type::Redo ) );
    return res;
}

void RedoMenuItem::updateRedoListCache_( const HistoryStore& store, HistoryStore::ChangeType )
{
    auto lastRedos = store.getNActions( cMaxNumActionsInList, HistoryAction::Type::Redo );
    if ( lastRedos.empty() )
    {
        dropList_.clear();
        return;
    }

    dropList_.resize( lastRedos.size() );
    for ( int i = 0; i < lastRedos.size(); ++i )
    {
        dropList_[i] = std::make_shared<LambdaRibbonItem>( lastRedos[i] + "##" + std::to_string( i ),
            [i] ()
        {
            if ( !HistoryStore::getViewerInstance() )
                return;
            for ( int j = 0; j <= i; ++j )
                HistoryStore::getViewerInstance()->redo();
        } );
    }
}

MR_REGISTER_RIBBON_ITEM( UndoMenuItem )

MR_REGISTER_RIBBON_ITEM( RedoMenuItem )

}
