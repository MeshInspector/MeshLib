#include "MRMeshModifier.h"
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRString.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"

namespace MR
{

MeshModifier::MeshModifier( std::string name, StatePluginTabs tab ):
    RibbonMenuItem( std::move( name ) ),
    tab_{tab}
{
}

bool MeshModifier::action()
{
    auto objs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    modify( objs );
    return false;
}

bool MeshModifier::modify( const std::vector<std::shared_ptr<VisualObject>>& selectedObjects )
{    
    bool res = modify_( selectedObjects );

    for ( const auto& data : selectedObjects )
        data->setDirtyFlags( DIRTY_ALL );

    return res;
}

StatePluginTabs MeshModifier::getTab() const
{
    return tab_;
}

bool MeshModifier::checkStringMask( const std::string& mask ) const
{
    return findSubstringCaseInsensitive( name(), mask ) != std::string::npos;
}

}
