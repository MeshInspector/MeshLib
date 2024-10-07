#include "MRSceneOperations.h"

#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectsAccess.h"

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

} // namespace MR
