#pragma once
#include "MRHistoryAction.h"
#include "MRObjectPoints.h"
#include "MRPointCloud.h"
#include "MRHeapBytes.h"
#include <memory>


namespace MR
{

/// Undo action for changing normals in PointCloud
/// \ingroup HistoryGroup
class ChangePointCloudNormalsAction : public HistoryAction
{
public:
    using Obj = ObjectPoints;

    /// use this constructor to remember point cloud's normals before making any changes in them
    ChangePointCloudNormalsAction( std::string name, const std::shared_ptr<ObjectPoints>& obj ) :
        objPoints_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            if ( auto pc = obj->pointCloud() )
                backupNormals_ = pc->normals;
        }
    }

    /// use this constructor to remember point cloud's normals and immediate set new value
    ChangePointCloudNormalsAction( std::string name, const std::shared_ptr<ObjectPoints>& obj, VertNormals && newNormals ) :
        objPoints_{ obj },
        backupNormals_{ std::move( newNormals ) },
        name_{ std::move( name ) }
    {
        action( HistoryAction::Type::Redo );
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objPoints_ || !objPoints_->varPointCloud() )
            return;
        
        std::swap( objPoints_->varPointCloud()->normals, backupNormals_ );
        setObjectDirty( objPoints_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectPoints>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_RENDER_NORMALS );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + backupNormals_.heapBytes();
    }

private:
    std::shared_ptr<ObjectPoints> objPoints_;
    VertNormals backupNormals_;

    std::string name_;
};

}