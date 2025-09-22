#pragma once
#include "MRHistoryAction.h"
#include "MRObjectLines.h"
#include "MRPolyline.h"
#include "MRHeapBytes.h"
#include <memory>

namespace MR
{

/// \defgroup HistoryGroup History group
/// \{

/// Undo action for ObjectLines polyline change
class ChangePolylineAction : public HistoryAction
{
public:
    using Obj = ObjectLines;

    /// use this constructor to remember object's polyline before making any changes in it
    ChangePolylineAction( std::string name, const std::shared_ptr<ObjectLines>& obj ) :
        objLines_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            if ( auto p = obj->polyline() )
                clonePolyline_ = std::make_shared<Polyline3>( *p );
        }
    }

    /// use this constructor to remember object's polyline and immediately set new polyline
    ChangePolylineAction( std::string name, const std::shared_ptr<ObjectLines>& obj, std::shared_ptr<Polyline3> newPolyline ) :
        objLines_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
            clonePolyline_ = obj->updatePolyline( std::move( newPolyline ) );
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objLines_ )
            return;

        clonePolyline_ = objLines_->updatePolyline( clonePolyline_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectLines>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_ALL );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + MR::heapBytes( clonePolyline_ );
    }

private:
    std::shared_ptr<ObjectLines> objLines_;
    std::shared_ptr<Polyline3> clonePolyline_;

    std::string name_;
};

/// Undo action for ObjectLines points only (not topology) change
class ChangePolylinePointsAction : public HistoryAction
{
public:
    using Obj = ObjectLines;

    /// use this constructor to remember object's lines points before making any changes in it
    ChangePolylinePointsAction( std::string name, const std::shared_ptr<ObjectLines>& obj ) :
        objLines_{ obj },
        name_{ std::move( name ) }
    {
        if ( !objLines_ )
            return;
        if ( auto p = objLines_->polyline() )
            clonePoints_ = p->points;
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objLines_ )
            return;

        if ( auto p = objLines_->varPolyline() )
        {
            std::swap( p->points, clonePoints_ );
            objLines_->setDirtyFlags( DIRTY_POSITION );
        }
    }

    static void setObjectDirty( const std::shared_ptr<ObjectLines>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_POSITION );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + clonePoints_.heapBytes();
    }

private:
    std::shared_ptr<ObjectLines> objLines_;
    VertCoords clonePoints_;

    std::string name_;
};

/// Undo action for ObjectLines topology only (not points) change
class ChangePolylineTopologyAction : public HistoryAction
{
public:
    using Obj = ObjectLines;

    /// use this constructor to remember object's lines points before making any changes in it
    ChangePolylineTopologyAction( std::string name, const std::shared_ptr<ObjectLines>& obj ) :
        objLines_{ obj },
        name_{ std::move( name ) }
    {
        if ( !objLines_ )
            return;
        if ( auto p = objLines_->polyline() )
            cloneTopology_ = p->topology;
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objLines_ )
            return;

        if ( auto p = objLines_->varPolyline() )
        {
            std::swap( p->topology, cloneTopology_ );
            objLines_->setDirtyFlags( DIRTY_FACE );
        }
    }

    static void setObjectDirty( const std::shared_ptr<ObjectLines>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_FACE );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + cloneTopology_.heapBytes();
    }

private:
    std::shared_ptr<ObjectLines> objLines_;
    PolylineTopology cloneTopology_;

    std::string name_;
};


/// Undo action that modifies one point's coordinates inside ObjectPolyline
/// \ingroup HistoryGroup
class ChangeOnePointInPolylineAction : public HistoryAction
{
public:
    using Obj = ObjectLines;

    /// use this constructor to remember point's coordinates before making any changes in it
    ChangeOnePointInPolylineAction( std::string name, const std::shared_ptr<ObjectLines>& obj, VertId pointId ) :
        objPoints_{ obj },
        pointId_{ pointId },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            if ( auto m = obj->polyline() )
                if ( m->points.size() > pointId_ )
                    safeCoords_ = m->points[pointId_];
        }
    }

    /// use this constructor to remember point's coordinates and immediate set new coordinates
    ChangeOnePointInPolylineAction( std::string name, const std::shared_ptr<ObjectLines>& obj, VertId pointId, const Vector3f & newCoords ) :
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

        if ( auto m = objPoints_->varPolyline() )
        {
            if ( m->points.size() > pointId_ )
            {
                std::swap( safeCoords_, m->points[pointId_] );
                objPoints_->setDirtyFlags( DIRTY_POSITION );
            }
        }
    }

    static void setObjectDirty( const std::shared_ptr<ObjectLines>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_POSITION );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::shared_ptr<ObjectLines> objPoints_;
    VertId pointId_;
    Vector3f safeCoords_;

    std::string name_;
};

/// \}

} // namespace MR
