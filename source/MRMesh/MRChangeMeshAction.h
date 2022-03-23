#pragma once
#include "MRHistoryAction.h"
#include "MRObjectMesh.h"
#include "MRMesh.h"
#include <memory>

namespace MR
{

// Undo action for ObjectMesh mesh change
class ChangeMeshAction : public HistoryAction
{
public:
    // use this constructor to remember object's mesh before making any changes in it
    ChangeMeshAction( std::string name, const std::shared_ptr<ObjectMesh>& obj ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            if ( auto m = obj->mesh() )
                cloneMesh_ = std::make_shared<Mesh>( *m );
        }
    }

    // use this constructor to remember current object's mesh and immediately set new mesh to the object
    ChangeMeshAction( std::string name, const std::shared_ptr<ObjectMesh>& obj, std::shared_ptr<Mesh> newMesh ) :
        objMesh_{ obj },
        cloneMesh_{ std::move( newMesh ) },
        name_{ std::move( name ) }
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

// Undo action for ObjectMesh points only (not topology) change
class ChangeMeshPointsAction : public HistoryAction
{
public:
    // use this constructor to remember object's mesh points before making any changes in it
    ChangeMeshPointsAction( std::string name, const std::shared_ptr<ObjectMesh>& obj ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( !objMesh_ )
            return;
        if ( auto m = objMesh_->mesh() )
            clonePoints_ = m->points;
    }

    // use this constructor to remember current object's mesh points and immediately set new mesh points to the object
    ChangeMeshPointsAction( std::string name, const std::shared_ptr<ObjectMesh>& obj, VertCoords newPoints ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( !objMesh_ )
            return;
        if ( auto m = objMesh_->varMesh() )
        {
            clonePoints_ = m->points;
            m->points = std::move( newPoints );
            objMesh_->setDirtyFlags( DIRTY_POSITION );
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

        if ( auto m = objMesh_->varMesh() )
        {
            std::swap( m->points, clonePoints_ );
            objMesh_->setDirtyFlags( DIRTY_POSITION );
        }
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    VertCoords clonePoints_;

    std::string name_;
};

// Undo action for ObjectMesh topology only (not points) change
class ChangeMeshTopologyAction : public HistoryAction
{
public:
    // use this constructor to remember object's mesh points before making any changes in it
    ChangeMeshTopologyAction( std::string name, const std::shared_ptr<ObjectMesh>& obj ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( !objMesh_ )
            return;
        if ( auto m = objMesh_->mesh() )
            cloneTopology_ = m->topology;
    }

    // use this constructor to remember current object's mesh topology and immediately set new mesh topology to the object
    ChangeMeshTopologyAction( std::string name, const std::shared_ptr<ObjectMesh>& obj, MeshTopology newTopology ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( !objMesh_ )
            return;
        if ( auto m = objMesh_->varMesh() )
        {
            cloneTopology_ = m->topology;
            m->topology = std::move( newTopology );
            objMesh_->setDirtyFlags( DIRTY_FACE );
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

        if ( auto m = objMesh_->varMesh() )
        {
            std::swap( m->topology, cloneTopology_ );
            objMesh_->setDirtyFlags( DIRTY_FACE );
        }
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    MeshTopology cloneTopology_;

    std::string name_;
};

} // namespace MR
