#pragma once
#include "MRHistoryAction.h"
#include "MRObjectPoints.h"
#include "MRPointCloud.h"
#include "MRHeapBytes.h"
#include <memory>


namespace MR
{

/// Undo action for ObjectMesh mesh change
/// \ingroup HistoryGroup
class ChangePointCloudAction : public HistoryAction
{
public:
    using Obj = ObjectPoints;

    /// use this constructor to remember object's point cloud before making any changes in it
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

/// Undo action for points field inside ObjectPoints
/// \ingroup HistoryGroup
class ChangePointCloudPointsAction : public HistoryAction
{
public:
    using Obj = ObjectPoints;

    /// use this constructor to remember object's points field before making any changes in it
    ChangePointCloudPointsAction( std::string name, const std::shared_ptr<ObjectPoints>& obj ) :
        objPoints_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            if ( auto m = obj->pointCloud() )
                clonePoints_ = m->points;
        }
    }

    /// use this constructor to remember object's points field and immediate set new value
    ChangePointCloudPointsAction( std::string name, const std::shared_ptr<ObjectPoints>& obj, VertCoords && newPoints ) :
        objPoints_{ obj },
        clonePoints_{ std::move( newPoints ) },
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
        if ( !objPoints_ )
            return;

        if ( auto m = objPoints_->varPointCloud() )
        {
            std::swap( m->points, clonePoints_ );
            objPoints_->setDirtyFlags( DIRTY_POSITION );
        }
    }

    static void setObjectDirty( const std::shared_ptr<ObjectPoints>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_POSITION );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + clonePoints_.heapBytes();
    }

private:
    std::shared_ptr<ObjectPoints> objPoints_;
    VertCoords clonePoints_;

    std::string name_;
};

/// Undo action that modifies one point's coordinates inside ObjectPoints
/// \ingroup HistoryGroup
class ChangeOnePointInCloudAction : public HistoryAction
{
public:
    using Obj = ObjectPoints;

    /// use this constructor to remember point's coordinates before making any changes in it
    ChangeOnePointInCloudAction( std::string name, const std::shared_ptr<ObjectPoints>& obj, VertId pointId ) :
        objPoints_{ obj },
        pointId_{ pointId },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            if ( auto m = obj->pointCloud() )
                if ( m->points.size() > pointId_ )
                    safeCoords_ = m->points[pointId_];
        }
    }

    /// use this constructor to remember point's coordinates and immediate set new coordinates
    ChangeOnePointInCloudAction( std::string name, const std::shared_ptr<ObjectPoints>& obj, VertId pointId, const Vector3f & newCoords ) :
        objPoints_{ obj },
        pointId_{ pointId },
        safeCoords_{ newCoords },
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
        if ( !objPoints_ )
            return;

        if ( auto m = objPoints_->varPointCloud() )
        {
            if ( m->points.size() > pointId_ )
            {
                std::swap( safeCoords_, m->points[pointId_] );
                objPoints_->setDirtyFlags( DIRTY_POSITION );
            }
        }
    }

    static void setObjectDirty( const std::shared_ptr<ObjectPoints>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_POSITION );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::shared_ptr<ObjectPoints> objPoints_;
    VertId pointId_;
    Vector3f safeCoords_;

    std::string name_;
};

} //namespace MR
