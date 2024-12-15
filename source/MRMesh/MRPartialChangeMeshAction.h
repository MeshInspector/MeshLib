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
    /// use this constructor to set new object mesh and remember its difference from existed mesh for future undoing
    PartialChangeMeshAction( std::string name, const std::shared_ptr<ObjectMesh>& obj, std::shared_ptr<Mesh>&& newMesh ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        assert( obj && newMesh );
        if ( obj )
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

/// \}

} // namespace MR
