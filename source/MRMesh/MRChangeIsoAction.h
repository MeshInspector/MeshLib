#pragma once
#ifndef __EMSCRIPTEN__
#include "MRHistoryAction.h"
#include "MRObjectVoxels.h"
#include "MRMesh.h"
#include "MRHeapBytes.h"
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
    /// use this constructor to remember object's surface and iso before making any changes in it
    ChangeIsoAction( std::string name, const std::shared_ptr<ObjectVoxels>& obj ) :
        objVoxels_{ obj },
        name_{ std::move( name ) }
    {
        if ( obj )
        {
            if ( auto m = obj->mesh() )
                cloneSurface_ = std::make_shared<Mesh>( *m );
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
    float storedIso_{ 0.0f };

    std::string name_;
};

/// \}

}

#endif