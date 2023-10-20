#pragma once
#include "MRHistoryAction.h"
#include "MRObjectMeshHolder.h"
#include "MRObjectLinesHolder.h"
#include <memory>
#include "MRChangeVertsColorMapAction.h"

namespace MR
{

/// History action for texture change
/// \ingroup HistoryGroup
class ChangeTextureAction : public HistoryAction
{
public:
    using Obj = ObjectMeshHolder;
    /// Constructed from original obj
    ChangeTextureAction( const std::string& name, const std::shared_ptr<ObjectMeshHolder>& obj ) :
        obj_{ obj },
        name_{ name }
    {
        if ( obj )
            texture_ = obj->getTexture();
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        obj_->updateTexture( texture_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectMeshHolder>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_TEXTURE );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + texture_.heapBytes();
    }

private:
    std::shared_ptr<ObjectMeshHolder> obj_;
    MeshTexture texture_;
    std::string name_;
};

/// History action for faces color map change
/// \ingroup HistoryGroup
class ChangeFacesColorMapAction : public HistoryAction
{
public:
    using Obj = ObjectMeshHolder;
    /// Constructed from original obj
    ChangeFacesColorMapAction( const std::string& name, const std::shared_ptr<ObjectMeshHolder>& obj ) :
        obj_{ obj },
        name_{ name }
    {
        if ( obj )
            colorMap_ = obj->getFacesColorMap();
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        obj_->updateFacesColorMap( colorMap_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectMeshHolder>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_PRIMITIVE_COLORMAP );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + colorMap_.heapBytes();
    }

private:
    std::shared_ptr<ObjectMeshHolder> obj_;
    FaceColors colorMap_;
    std::string name_;
};

/// History action for lines color map change
/// \ingroup HistoryGroup
class ChangeLinesColorMapAction : public HistoryAction
{
public:
    using Obj = ObjectLinesHolder;
    /// Constructed from original obj
    ChangeLinesColorMapAction( const std::string& name, const std::shared_ptr<ObjectLinesHolder>& obj ) :
        obj_{ obj },
        name_{ name }
    {
        if ( obj )
            colorMap_ = obj->getLinesColorMap();
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        obj_->updateLinesColorMap( colorMap_ );
    }

    static void setObjectDirty( const std::shared_ptr<ObjectLinesHolder>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_PRIMITIVE_COLORMAP );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + colorMap_.heapBytes();
    }

private:
    std::shared_ptr<ObjectLinesHolder> obj_;
    UndirectedEdgeColors colorMap_;
    std::string name_;
};

}
