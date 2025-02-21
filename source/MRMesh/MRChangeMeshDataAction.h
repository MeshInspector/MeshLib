#pragma once

#include "MRHistoryAction.h"
#include "MRObjectMesh.h"

namespace MR
{

/// \defgroup HistoryGroup History group
/// \{

/// Undo action for ObjectMeshData change
class ChangeMeshDataAction : public HistoryAction
{
public:
    using Obj = ObjectMesh;

    /// use this constructor to remember object's data before making any changes in it
    ChangeMeshDataAction( std::string name, const std::shared_ptr<ObjectMesh>& obj ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( objMesh_ )
            data_ = objMesh_->data();
    }

    /// use this constructor to remember object's data and immediately set new data
    ChangeMeshDataAction( std::string name, const std::shared_ptr<ObjectMesh>& obj, ObjectMeshData&& newData ) :
        objMesh_{ obj },
        name_{ std::move( name ) }
    {
        if ( objMesh_ )
        {
            data_ = std::move( newData );
            objMesh_->updateData( data_ );
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

        objMesh_->updateData( data_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectMesh>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_ALL );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + data_.heapBytes();
    }

private:
    std::shared_ptr<ObjectMesh> objMesh_;
    ObjectMeshData data_;

    std::string name_;
};

/// \}

} // namespace MR
