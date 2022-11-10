#pragma once
#ifndef __EMSCRIPTEN__
#include "MRHistoryAction.h"
#include "MRObjectVoxels.h"
#include "MRMesh.h"
#include "MRHeapBytes.h"
#include "MRFloatGrid.h"
#include <memory>

namespace MR
{

/// \defgroup HistoryGroup History group
/// \{

/// Undo action for ObjectVoxels iso change
class ChangeIsoAction : public HistoryAction
{
public:
    using Obj = ObjectVoxels;
    /// use this constructor to remember object's iso before making any changes in it
    ChangeIsoAction( std::string name, const std::shared_ptr<ObjectVoxels>& obj ) :
        objVoxels_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            storedIso_ = obj->getIsoValue();
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objVoxels_ )
            return;

        float newIso = objVoxels_->getIsoValue();
        objVoxels_->setIsoValue( storedIso_, {}, false );
        storedIso_ = newIso;
    }

    static void setObjectDirty( const std::shared_ptr<Obj>& )
    {
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::shared_ptr<ObjectVoxels> objVoxels_;
    float storedIso_{ 0.0f };

    std::string name_;
};

// Undo action for ObjectVoxels active bounds change
class ChangeActiveBoxAction : public HistoryAction
{
public:
    using Obj = ObjectVoxels;
    /// use this constructor to remember object's active box before making any changes in it
    ChangeActiveBoxAction( std::string name, const std::shared_ptr<ObjectVoxels>& obj ) :
        objVoxels_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            activeBox_ = obj->getActiveBounds();
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objVoxels_ )
            return;

        auto box = objVoxels_->getActiveBounds();
        objVoxels_->setActiveBounds( activeBox_, {}, false );
        activeBox_ = box;
    }

    static void setObjectDirty( const std::shared_ptr<Obj>& )
    {
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::shared_ptr<ObjectVoxels> objVoxels_;
    Box3i activeBox_;

    std::string name_;
};

// Undo action for ObjectVoxels surface change (need for faster undo redo)
class ChangeSurfaceAction : public HistoryAction
{
public:
    using Obj = ObjectVoxels;
    /// use this constructor to remember object's surface before making any changes in it
    ChangeSurfaceAction( std::string name, const std::shared_ptr<ObjectVoxels>& obj ) :
        objVoxels_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            if ( auto m = obj->mesh() )
                cloneSurface_ = std::make_shared<Mesh>( *m );
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objVoxels_ )
            return;

        cloneSurface_ = objVoxels_->updateIsoSurface( cloneSurface_ );
    }

    static void setObjectDirty( const std::shared_ptr<Obj>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_ALL );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + MR::heapBytes( cloneSurface_ );
    }

private:
    std::shared_ptr<ObjectVoxels> objVoxels_;
    std::shared_ptr<Mesh> cloneSurface_;

    std::string name_;
};

// Undo action for ObjectVoxels surface change (need for faster undo redo)
class ChangeGridAction : public HistoryAction
{
public:
    using Obj = ObjectVoxels;
    /// use this constructor to remember object's surface before making any changes in it
    ChangeGridAction( std::string name, const std::shared_ptr<ObjectVoxels>& obj ) :
    objVoxels_{ obj },
    name_{ std::move( name ) }
    {
        if ( obj )
        {
            grid_ = obj->grid();
            voxelSize_ = obj->voxelSize();
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objVoxels_ )
            return;

        objVoxels_->construct( grid_, voxelSize_ );
    }

    static void setObjectDirty( const std::shared_ptr<Obj>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_ALL );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + ( grid_ ? grid_->memUsage() : 0);
    }

private:
    std::shared_ptr<ObjectVoxels> objVoxels_;
    FloatGrid grid_;
    Vector3f voxelSize_;

    std::string name_;
};

/// \}

}

#endif