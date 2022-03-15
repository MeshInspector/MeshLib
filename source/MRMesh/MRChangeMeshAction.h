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
    // use this constructor to remember object's mesh before making any changes in it
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

    // use this constructor to remember current object's mesh and immediately set new mesh to the object
    ChangeMeshAction( const std::string& name, const std::shared_ptr<ObjectMesh>& obj, std::shared_ptr<Mesh> newMesh ) :
        objMesh_{ obj },
        cloneMesh_{ std::move( newMesh ) },
        name_{ name }
    {
        if ( objMesh_ )
            objMesh_->swapMesh( cloneMesh_);
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