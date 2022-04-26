#pragma once
#include "MRHistoryAction.h"
#include "MRObjectPoints.h"
#include "MRPointCloud.h"
#include "MRHeapBytes.h"
#include <memory>


namespace MR
{

// Undo action for ObjectMesh mesh change
class ChangePointCloudAction : public HistoryAction
{
public:
    using Obj = ObjectPoints;

    // use this constructor to remember object's point cloud before making any changes in it
    ChangePointCloudAction( std::string name, const std::shared_ptr<ObjectPoints>& obj ) :
        objPoints_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            if ( auto m = obj->pointCloud() )
                clonePointCloud_ = std::make_shared<PointCloud>( *m );
        }
    }

    virtual std::string name() const override { return name_; }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objPoints_ )
            return;

        objPoints_->swapPointCloud( clonePointCloud_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectPoints>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_ALL );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    { return name_.capacity() + MR::heapBytes( clonePointCloud_ ); }

private:
    std::shared_ptr<ObjectPoints> objPoints_;
    std::shared_ptr<PointCloud> clonePointCloud_;

    std::string name_;
};

}
