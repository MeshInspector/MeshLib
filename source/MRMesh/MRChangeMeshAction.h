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
        name_{ name }
    {
        if ( obj )
        {
            if ( auto m = obj->mesh() )
                cloneMesh_ = std::make_shared<Mesh>( *m );
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objMesh_ )
            return;

        objMesh_->swapMesh( cloneMesh_);
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    std::shared_ptr<Mesh> cloneMesh_;

    std::string name_;
};
}