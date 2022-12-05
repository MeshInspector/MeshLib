#include "MRRibbonSceneButtons.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRChangeSceneObjectsOrder.h"
#include "MRMesh/MRChangeSceneAction.h"

namespace
{

// select and show only neighborhood. If `selectNext` true - select next, otherwise select prev
void selectAndShowOnlyNeighborhood( bool selectNext = true )
{
    using namespace MR;
    const auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
    auto currentObjs = getTopmostVisibleObjects( &SceneRoot::get(), ObjectSelectivityType::Selectable );
    currentObjs.insert( currentObjs.begin(), selected.begin(), selected.end() );
    std::shared_ptr<Object> first = !currentObjs.empty() ? currentObjs.front() : std::shared_ptr<Object>{};
    if ( !first )
    {
        // nothing is selected or visible, just select and show the first object
        first = getDepthFirstObject( &SceneRoot::get(), ObjectSelectivityType::Selectable );
    }
    Object* newSelection = first.get();
    if ( first )
    {
        // chose a sibling object of first
        if ( const auto firstParent = first->parent() )
        {
            const auto& firstParentChildren = firstParent->children();
            if ( firstParentChildren.size() > 1 )
            {
                // include all siblings in currentObjs to hide and de-select them
                currentObjs.insert( currentObjs.end(), firstParentChildren.begin(), firstParentChildren.end() );
                if ( !selectNext )
                {
                    newSelection = firstParentChildren.back().get();
                    for ( const auto& child : firstParentChildren )
                    {
                        if ( child == first )
                            break;
                        newSelection = child.get();
                    }
                }
                else
                {
                    // select previous
                    newSelection = firstParentChildren.front().get();
                    for ( auto it = firstParentChildren.rbegin(); it != firstParentChildren.rend(); ++it )
                    {
                        if ( *it == first )
                            break;
                        newSelection = it->get();
                    }
                }
            }
        }
    }
    if ( newSelection )
    {
        for ( const auto& obj : currentObjs )
        {
            obj->setVisible( false );
            obj->select( false );
        }
        newSelection->select( true );
        newSelection->setGlobalVisibilty( true );
    }
}

}

namespace MR
{

RibbonSceneSortByName::RibbonSceneSortByName() :
    RibbonMenuItem( "Ribbon Scene Sort by name" )
{
}

std::string RibbonSceneSortByName::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    if ( SceneRoot::get().children().empty() )
        return "At least one objects should be in scene";
    return "";
}

bool RibbonSceneSortByName::action()
{
    SCOPED_HISTORY( "Sort scene" );
    sortObjectsRecursive_( SceneRoot::getSharedPtr() );
    return false;
}

void RibbonSceneSortByName::sortObjectsRecursive_( std::shared_ptr<Object> object )
{
    auto& children = object->children();
    for ( const auto& child : children )
        sortObjectsRecursive_( child );

    AppendHistory( std::make_shared<ChangeSceneObjectsOrder>( "Sort object children", object ) );
    object->sortChildren();
}

RibbonSceneSelectAll::RibbonSceneSelectAll() :
    RibbonMenuItem("Ribbon Scene Select all")
{
}

bool RibbonSceneSelectAll::action()
{
    const auto selectable = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selectable );
    for ( auto obj : selectable )
    {
        obj->select( true );
        obj->setVisible( true );
    }
    return false;
}

RibbonSceneUnselectAll::RibbonSceneUnselectAll() :
    RibbonMenuItem( "Ribbon Scene Unselect all" )
{
}

bool RibbonSceneUnselectAll::action()
{
    const auto selectable = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selectable );
    for ( auto obj : selectable )
        obj->select( false );
    return false;
}

RibbonSceneShowOnlyPrev::RibbonSceneShowOnlyPrev() :
    RibbonMenuItem( "Ribbon Scene Show only previous" )
{
}

std::string RibbonSceneShowOnlyPrev::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    if ( SceneRoot::get().children().empty() )
        return "At least one objects should be in scene";
    return "";
}

bool RibbonSceneShowOnlyPrev::action()
{
    selectAndShowOnlyNeighborhood( false );
    return false;
}

RibbonSceneShowOnlyNext::RibbonSceneShowOnlyNext() :
    RibbonMenuItem( "Ribbon Scene Show only next" )
{
}

std::string RibbonSceneShowOnlyNext::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    if ( SceneRoot::get().children().empty() )
        return "At least one objects should be in scene";
    return "";
}

bool RibbonSceneShowOnlyNext::action()
{
    selectAndShowOnlyNeighborhood();
    return false;
}

RibbonSceneRename::RibbonSceneRename() :
    RibbonMenuItem( "Ribbon Scene Rename" )
{
}

bool RibbonSceneRename::action()
{
    getViewerInstance().getMenuPluginAs<RibbonMenu>()->tryRenameSelectedObject();
    return false;
}

RibbonSceneRemoveSelected::RibbonSceneRemoveSelected() :
    RibbonMenuItem( "Ribbon Scene Remove selected objects" )
{
}

std::string RibbonSceneRemoveSelected::isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const
{
    auto res = SceneStateAtLeastCheck<1, Object>::isAvailable( objs );
    auto allowRemoval = getViewerInstance().getMenuPluginAs<RibbonMenu>()->checkPossibilityObjectRemoval();
    if ( !allowRemoval )
    {
        if ( !res.empty() )
            res += "\n";
        res += "Deleting objects is blocked";
    }
    return res;
}

bool RibbonSceneRemoveSelected::action()
{
    auto allowRemoval = getViewerInstance().getMenuPluginAs<RibbonMenu>()->checkPossibilityObjectRemoval();
    if ( !allowRemoval )
        return false;

    const auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
    SCOPED_HISTORY( "Remove objects" );
    for ( int i = (int) selected.size() - 1; i >= 0; --i )
        if ( selected[i] )
        {
            // for now do it by one object
            AppendHistory<ChangeSceneAction>( "Remove object", selected[i], ChangeSceneAction::Type::RemoveObject );
            selected[i]->detachFromParent();
        }
    return false;
}

MR_REGISTER_RIBBON_ITEM( RibbonSceneSortByName )
MR_REGISTER_RIBBON_ITEM( RibbonSceneSelectAll )
MR_REGISTER_RIBBON_ITEM( RibbonSceneUnselectAll )
MR_REGISTER_RIBBON_ITEM( RibbonSceneShowOnlyPrev )
MR_REGISTER_RIBBON_ITEM( RibbonSceneShowOnlyNext )
MR_REGISTER_RIBBON_ITEM( RibbonSceneRename )
MR_REGISTER_RIBBON_ITEM( RibbonSceneRemoveSelected )

}
