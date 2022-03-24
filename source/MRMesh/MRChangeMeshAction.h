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
    using Obj = ObjectMesh;

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

    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_ALL );
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
    using Obj = ObjectMesh;

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

    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_POSITION );
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
    using Obj = ObjectMesh;

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

    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_FACE );
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    MeshTopology cloneTopology_;

    std::string name_;
};

} // namespace MR
