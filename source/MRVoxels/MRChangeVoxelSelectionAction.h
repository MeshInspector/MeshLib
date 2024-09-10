#pragma once
#include "MRVoxelsFwd.h"

#include "MRMesh/MRHistoryAction.h"
#include "MRObjectVoxels.h"

namespace MR
{
/// \addtogroup HistoryGroup
/// \{

/// Undo action for ObjectVoxels face selection
class ChangVoxelSelectionAction : public HistoryAction
{
public:
    using Obj = ObjectVoxels;

    /// use this constructor to remember object's face selection before making any changes in it
    ChangVoxelSelectionAction( const std::string& name, const std::shared_ptr<Obj>& objVoxels ) :
        name_{ name },
        objVoxels_{ objVoxels }
    {
        if ( !objVoxels_ )
            return;
        selection_ = objVoxels_->getSelectedVoxels();
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( Type ) override
    {
        if ( !objVoxels_ )
            return;
        auto tmp = objVoxels_->getSelectedVoxels();
        objVoxels_->selectVoxels( selection_ );
        selection_ = std::move( tmp );
    }

    const VoxelBitSet& selection() const
    {
        return selection_;
    }

    /// empty because set dirty is inside selectFaces
    static void setObjectDirty( const std::shared_ptr<Obj>& )
    {}

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + selection_.heapBytes();
    }

private:
    std::string name_;
    std::shared_ptr<Obj> objVoxels_;
    VoxelBitSet selection_;
};

}
