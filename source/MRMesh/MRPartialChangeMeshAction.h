#pragma once

#include "MRHistoryAction.h"
#include "MRObjectMesh.h"
#include "MRMeshDiff.h"
#include <cassert>

namespace MR
{

/// \defgroup HistoryGroup History group
/// \{

/// Undo action for efficiently storage of partial change in mesh (e.g. a modification of small region)
class PartialChangeMeshAction : public HistoryAction
{
public:
    /// use this constructor to set new object's mesh and remember its difference from existed mesh for future undoing
    PartialChangeMeshAction( std::string name, std::shared_ptr<ObjectMesh> obj, std::shared_ptr<Mesh>&& newMesh ) :
        objMesh_{ std::move( obj ) },
        name_{ std::move( name ) }
    {
        assert( objMesh_ && newMesh );
        if ( objMesh_ )
        {
            auto oldMesh = objMesh_->updateMesh( std::move( newMesh ) );
            if ( oldMesh && objMesh_->mesh() )
                meshDiff_ = MeshDiff( *objMesh_->mesh(), *oldMesh );
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

        auto m = objMesh_->varMesh();
        assert( m );
        if ( !m )
            return;

        meshDiff_.applyAndSwap( *m );
        objMesh_->setDirtyFlags( DIRTY_ALL );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + meshDiff_.heapBytes();
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    MeshDiff meshDiff_;

    std::string name_;
};

/// Undo action for efficiently storage of partial change in mesh points (e.g. a modification of small region)
class PartialChangeMeshPointsAction : public HistoryAction
{
public:
    /// use this constructor to set new object's points and remember its difference from existed points for future undoing
    PartialChangeMeshPointsAction( std::string name, std::shared_ptr<ObjectMesh> obj, VertCoords&& newPoints ) :
        objMesh_{ std::move( obj ) },
        name_{ std::move( name ) }
    {
        assert( objMesh_ );
        if ( !objMesh_ )
            return;

        if ( auto m = objMesh_->varMesh() )
        {
            pointsDiff_ = VertCoordsDiff( newPoints, m->points );
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

        auto m = objMesh_->varMesh();
        assert( m );
        if ( !m )
            return;

        pointsDiff_.applyAndSwap( m->points );
        objMesh_->setDirtyFlags( DIRTY_POSITION );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + pointsDiff_.heapBytes();
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    VertCoordsDiff pointsDiff_;

    std::string name_;
};

/// Undo action for efficiently storage of partial change in mesh topology (e.g. a modification of small region)
class PartialChangeMeshTopologyAction : public HistoryAction
{
public:
    /// use this constructor to set new object's topology and remember its difference from existed topology for future undoing
    PartialChangeMeshTopologyAction( std::string name, std::shared_ptr<ObjectMesh> obj, MeshTopology&& newTopology ) :
        objMesh_{ std::move( obj ) },
        name_{ std::move( name ) }
    {
        assert( objMesh_ );
        if ( !objMesh_ )
            return;

        if ( auto m = objMesh_->varMesh() )
        {
            topologyDiff_ = MeshTopologyDiff( newTopology, m->topology );
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

        auto m = objMesh_->varMesh();
        assert( m );
        if ( !m )
            return;

        topologyDiff_.applyAndSwap( m->topology );
        objMesh_->setDirtyFlags( DIRTY_FACE );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + topologyDiff_.heapBytes();
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    MeshTopologyDiff topologyDiff_;

    std::string name_;
};

/// \}

} // namespace MR
