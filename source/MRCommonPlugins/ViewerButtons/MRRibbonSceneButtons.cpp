#include "MRRibbonSceneButtons.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewer/MRRibbonSchema.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRSceneObjectsListDrawer.h"
#include "MRViewer/ImGuiMenu.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRChangeSceneObjectsOrder.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRViewer/MRSceneObjectsListDrawer.h"

namespace MR
{

RibbonSceneSortByName::RibbonSceneSortByName() :
    RibbonMenuItem( "Ribbon Scene Sort by name" )
{
}

std::string RibbonSceneSortByName::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    if ( !getDepthFirstObject( &SceneRoot::get(), ObjectSelectivityType::Selectable ) )
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

std::string RibbonSceneSelectAll::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    if ( !getDepthFirstObject( &SceneRoot::get(), ObjectSelectivityType::Selectable ) )
        return "At least one objects should be in scene";
    return "";
}

bool RibbonSceneSelectAll::action()
{
    if ( auto menu = getViewerInstance().getMenuPlugin() )
    {
        if ( auto sceneList = menu->getSceneObjectsList() )
            sceneList->selectAllObjects();
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
    if ( !getDepthFirstObject( &SceneRoot::get(), ObjectSelectivityType::Selectable ) )
        return "At least one objects should be in scene";
    return "";
}

bool RibbonSceneShowOnlyPrev::action()
{
    auto menu = getViewerInstance().getMenuPlugin();
    if ( menu )
    {
        if ( auto sceneList = menu->getSceneObjectsList() )
            sceneList->changeVisible( false );
    }
    return false;
}

RibbonSceneShowOnlyNext::RibbonSceneShowOnlyNext() :
    RibbonMenuItem( "Ribbon Scene Show only next" )
{
}

std::string RibbonSceneShowOnlyNext::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    if ( !getDepthFirstObject( &SceneRoot::get(), ObjectSelectivityType::Selectable ) )
        return "At least one objects should be in scene";
    return "";
}

bool RibbonSceneShowOnlyNext::action()
{
    auto menu = getViewerInstance().getMenuPlugin();
    if ( menu )
    {
        if ( auto sceneList = menu->getSceneObjectsList() )
            sceneList->changeVisible( true );
    }
    return false;
}

RibbonSceneRename::RibbonSceneRename() :
    RibbonMenuItem( "Ribbon Scene Rename" )
{
}

bool RibbonSceneRename::action()
{
    getViewerInstance().getMenuPlugin()->tryRenameSelectedObject();
    return false;
}

RibbonSceneRemoveSelected::RibbonSceneRemoveSelected() :
    RibbonMenuItem( "Ribbon Scene Remove selected objects" )
{
}

std::string RibbonSceneRemoveSelected::isAvailable( const std::vector<std::shared_ptr<const Object>>& objs ) const
{
    auto res = SceneStateAtLeastCheck<1, Object, NoModelCheck>::isAvailable( objs );
    auto allowRemoval = getViewerInstance().getMenuPlugin()->checkPossibilityObjectRemoval();
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
    if ( auto menu = getViewerInstance().getMenuPlugin() )
        if ( !menu->checkPossibilityObjectRemoval() )
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
