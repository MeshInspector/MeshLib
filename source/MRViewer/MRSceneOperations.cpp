#include "MRSceneOperations.h"

#include "MRAppendHistory.h"
#include "MRRibbonMenu.h"
#include "MRSceneReorder.h"

#include "MRMesh/MRChangeSceneAction.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRTimer.h"

#include <unordered_set>

namespace MR
{

TypedFlatTree TypedFlatTree::fromFlatTree( const FlatTree& tree )
{
    std::vector<std::shared_ptr<ObjectMesh>> objsMesh;
    std::vector<std::shared_ptr<ObjectLines>> objsLines;
    std::vector<std::shared_ptr<ObjectPoints>> objsPoints;
    for ( const auto& subobj : tree.subobjects )
    {
        if ( auto objMesh = std::dynamic_pointer_cast<ObjectMesh>( subobj ) )
            objsMesh.emplace_back( std::move( objMesh ) );
        else if ( auto objLines = std::dynamic_pointer_cast<ObjectLines>( subobj ) )
            objsLines.emplace_back( std::move( objLines ) );
        else if ( auto objPoints = std::dynamic_pointer_cast<ObjectPoints>( subobj ) )
            objsPoints.emplace_back( std::move( objPoints ) );
    }
    return {
        .root = tree.root,
        .objsMesh = std::move( objsMesh ),
        .objsLines = std::move( objsLines ),
        .objsPoints = std::move( objsPoints ),
    };
}

std::vector<FlatTree> getFlatSubtrees( const std::vector<std::shared_ptr<Object>>& objs )
{
    std::unordered_set<Object *> objSet;
    objSet.reserve( objs.size() );
    for ( const auto& obj : objs )
        objSet.emplace( obj.get() );

    std::vector<FlatTree> results;
    for ( const auto& obj : objs )
    {
        // ignore if the object is a child of another object from the list
        for ( auto parent = obj->parent(); parent != nullptr; parent = parent->parent() )
            if ( objSet.contains( parent ) )
                continue;

        auto subobjs = getAllObjectsInTree( *obj );
        size_t found = 0;
        for ( const auto& subobj : subobjs )
            found += int( objSet.contains( subobj.get() ) );
        if ( found == 0 || found == subobjs.size() )
            results.emplace_back( FlatTree { obj, std::move( subobjs ) } );
    }
    return results;
}

/// moves all children (excluding meshes, points and lines) from one object to another
/// \return false if the move failed
static bool moveOtherChildrenWithUndo( Object& oldParent, Object& newParent )
{
    assert( &oldParent != &newParent );

    SceneReorder task
    {
        .to = &newParent,
    };
    for ( const auto& child : oldParent.children() )
    {
        if ( child->isAncillary() )
            continue;
        if ( dynamic_cast<ObjectMesh*>( child.get() ) )
            continue;
        if ( dynamic_cast<ObjectLines*>( child.get() ) )
            continue;
        if ( dynamic_cast<ObjectPoints*>( child.get() ) )
            continue;
        task.who.push_back( child.get() );
    }
    return sceneReorderWithUndo( task );
}

void mergeSubtree( TypedFlatTree subtree )
{
    MR_TIMER;

    auto& rootObj = subtree.root;
    assert( rootObj->parent() );

    auto& objsMesh = subtree.objsMesh;
    auto& objsLines = subtree.objsLines;
    auto& objsPoints = subtree.objsPoints;
    const auto objCount = objsMesh.size() + objsLines.size() + objsPoints.size();
    if ( objCount == 0 )
        return;

    if ( !objsMesh.empty() )
    {
        if ( auto rootObjMesh = std::dynamic_pointer_cast<ObjectMesh>( rootObj ) )
            objsMesh.insert( objsMesh.begin(), rootObjMesh );

        auto newObjMesh = merge( objsMesh );
        assert( newObjMesh );
        newObjMesh->setName( objsMesh.size() == objCount ? rootObj->name() : rootObj->name() + " (meshes)" );
        newObjMesh->select( true );

        AppendHistory<ChangeSceneAction>( "Add Object", newObjMesh, ChangeSceneAction::Type::AddObject );
        rootObj->parent()->addChild( newObjMesh );

        for ( const auto & objMesh : objsMesh )
            moveOtherChildrenWithUndo( *objMesh, *newObjMesh );
    }

    if ( !objsLines.empty() )
    {
        if ( auto rootObjLines = std::dynamic_pointer_cast<ObjectLines>( rootObj ) )
            objsLines.insert( objsLines.begin(), rootObjLines );

        auto newObjLines = merge( objsLines );
        assert( newObjLines );
        newObjLines->setName( objsLines.size() == objCount ? rootObj->name() : rootObj->name() + " (polylines)" );
        newObjLines->select( true );

        AppendHistory<ChangeSceneAction>( "Add Object", newObjLines, ChangeSceneAction::Type::AddObject );
        rootObj->parent()->addChild( newObjLines );

        for ( const auto & objLine : objsLines )
            moveOtherChildrenWithUndo( *objLine, *newObjLines );
    }

    if ( !objsPoints.empty() )
    {
        if ( auto rootObjPoints = std::dynamic_pointer_cast<ObjectPoints>( rootObj ) )
            objsPoints.insert( objsPoints.begin(), rootObjPoints );

        auto newObjPoints = merge( objsPoints );
        assert( newObjPoints );
        newObjPoints->setName( objsPoints.size() == objCount ? rootObj->name() : rootObj->name() + " (point clouds)" );
        newObjPoints->select( true );

        const auto hadNormals = std::any_of( objsPoints.begin(), objsPoints.end(), [] ( auto&& objPoints )
        {
            assert( objPoints );
            assert( objPoints->pointCloud() );
            return objPoints->pointCloud()->hasNormals();
        } );
        assert( newObjPoints->pointCloud() );
        if ( !newObjPoints->pointCloud()->hasNormals() && hadNormals )
        {
            pushNotification( {
                .text = "Some input point have normals and some others do not, all normals are lost",
                .type = NotificationType::Warning,
            } );
        }
        if ( newObjPoints->getRenderDiscretization() > 1 )
        {
            pushNotification( {
                .text = "Too many points in PointCloud:\nVisualization is simplified (only part of the points is drawn)",
                .type = NotificationType::Info,
            } );
        }

        AppendHistory<ChangeSceneAction>( "Add Object", newObjPoints, ChangeSceneAction::Type::AddObject );
        rootObj->parent()->addChild( newObjPoints );

        for ( const auto & objPoints : objsPoints )
            moveOtherChildrenWithUndo( *objPoints, *newObjPoints );
    }

    AppendHistory<ChangeSceneAction>( "Remove Object", rootObj, ChangeSceneAction::Type::RemoveObject );
    rootObj->detachFromParent();
}

void mergeSubtree( std::shared_ptr<Object> rootObj )
{
    return mergeSubtree( TypedFlatTree::fromFlatTree( FlatTree { rootObj, getAllObjectsInTree( *rootObj ) } ) );
}

} // namespace MR
