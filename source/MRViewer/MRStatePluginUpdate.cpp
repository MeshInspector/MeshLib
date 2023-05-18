#include "MRStatePluginUpdate.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectMesh.h"

namespace MR
{

void PluginCloseOnSelectedObjectRemove::onPluginEnable_()
{
    selectedObjs_ = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
}

void PluginCloseOnSelectedObjectRemove::onPluginDisable_()
{
    selectedObjs_.clear();
}

bool PluginCloseOnSelectedObjectRemove::shouldClose_() const
{
    for ( const auto& obj : selectedObjs_ )
    {
        if ( !obj->isAncestor( &SceneRoot::get() ) )
            return true;
    }
    return false;
}

void PluginCloseOnChangeMesh::onPluginEnable_()
{
    auto meshes = getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    meshChangedConnections_.reserve( meshes.size() );
    meshChanged_ = false;
    for ( auto& mesh : meshes )
        meshChangedConnections_.emplace_back( mesh->meshChangedSignal.connect( [&] ( uint32_t )
    {
        meshChanged_ = true;
    } ) );
}

void PluginCloseOnChangeMesh::onPluginDisable_()
{
    meshChangedConnections_.clear();
}

bool PluginCloseOnChangeMesh::shouldClose_() const
{
    return meshChanged_;
}

void PluginUpdateOnChangeMeshPart::preDrawUpdate()
{
    if ( !dirty_ || !func_ )
        return;
    func_();
    dirty_ = false;
}

void PluginUpdateOnChangeMeshPart::onPluginEnable_()
{
    dirty_ = true;
    auto meshes = getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    connections_.reserve( meshes.size() );
    for ( auto& mesh : meshes )
    {
        connections_.emplace_back( mesh->meshChangedSignal.connect( [&] ( uint32_t ) { dirty_ = true; } ) );
        connections_.emplace_back( mesh->faceSelectionChangedSignal.connect( [&] () { dirty_ = true; } ) );
    }
}

void PluginUpdateOnChangeMeshPart::onPluginDisable_()
{
    dirty_ = false;
    func_ = {};
    connections_.clear();
}

}