#pragma once
#include "MRHistoryAction.h"
#include "MRObjectMesh.h"
#include "MRMesh.h"
#include <memory>

namespace MR
{
// Change mesh action not to store extra Object information
class ChangeMeshAction : public HistoryAction
{
public:
    // Constructed from original ObjectMesh
    ChangeMeshAction( const std::string& name, const std::shared_ptr<ObjectMesh>& obj ) :
        objMesh_{ obj },
        mesh_{ obj->varMesh() },
        name_{ name }
    {
        cloneMesh_ = std::make_shared<Mesh>( *obj->varMesh() );
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objMesh_ || mesh_.expired() || !cloneMesh_ )
            return;

        auto mesh = mesh_.lock();
        std::swap( *mesh, *cloneMesh_ );
        objMesh_->setDirtyFlags( DIRTY_ALL );
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    std::weak_ptr<Mesh> mesh_;
    std::shared_ptr<Mesh> cloneMesh_;

    std::string name_;
};
}