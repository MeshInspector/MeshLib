#pragma once
#include "MRVoxelsFwd.h"

#include "MRMesh/MRHistoryAction.h"
#include "MRObjectVoxels.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRHeapBytes.h"
#include "MRFloatGrid.h"
#include <memory>

namespace MR
{

/// \defgroup HistoryGroup History group
/// \{

/// Undo action for ObjectVoxels iso-value change
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

/// Undo action for ObjectVoxels dual/standard marching cubes change
class ChangeDualMarchingCubesAction : public HistoryAction
{
public:
    using Obj = ObjectVoxels;
    /// use this constructor to remember object's dual-value before making any changes in it
    ChangeDualMarchingCubesAction( std::string name, const std::shared_ptr<ObjectVoxels>& obj ) :
        objVoxels_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            storedDual_ = obj->getDualMarchingCubes();
        }
    }

    /// use this constructor to remember given dual-value (and not the current value in the object)
    ChangeDualMarchingCubesAction( std::string name, const std::shared_ptr<ObjectVoxels>& obj, bool storeDual ) :
        objVoxels_{ obj },
        storedDual_( storeDual ),
        name_{ std::move( name ) }
    {
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !objVoxels_ )
            return;

        auto newDual = objVoxels_->getDualMarchingCubes();
        objVoxels_->setDualMarchingCubes( storedDual_, false );
        storedDual_ = newDual;
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
    bool storedDual_{ true };

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

// Undo action for ObjectVoxels all data change (need for faster undo redo)
class ChangeGridAction : public HistoryAction
{
public:
    using Obj = ObjectVoxels;
    /// use this constructor to remember object's data before making any changes in it
    ChangeGridAction( std::string name, const std::shared_ptr<ObjectVoxels>& obj ) :
    objVoxels_{ obj },    
    changeIsoAction_( name, obj ),
    changeSurfaceAction_(name, obj),
    name_{ std::move( name ) }
    {
        if ( obj )
        {
            vdbVolume_ = obj->vdbVolume();
            histogram_ = obj->histogram();
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type obj ) override
    {
        if ( !objVoxels_ )
            return;

        vdbVolume_ = objVoxels_->updateVdbVolume( std::move( vdbVolume_ ) );
        histogram_ = objVoxels_->updateHistogram( std::move( histogram_ ) );
        changeIsoAction_.action( obj );
        changeSurfaceAction_.action( obj );
    }

    static void setObjectDirty( const std::shared_ptr<Obj>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_ALL );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + histogram_.heapBytes() + changeIsoAction_.heapBytes() + changeSurfaceAction_.heapBytes() + MR::heapBytes( vdbVolume_.data );
    }

private:
    std::shared_ptr<ObjectVoxels> objVoxels_;
    VdbVolume vdbVolume_;
    Histogram histogram_;
    ChangeIsoAction changeIsoAction_;
    ChangeSurfaceAction changeSurfaceAction_;

    std::string name_;
};

/// \}

}
