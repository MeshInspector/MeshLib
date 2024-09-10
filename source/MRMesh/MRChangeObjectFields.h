#pragma once
#include "MRHistoryAction.h"
#include "MRVisualObject.h"
#include <memory>

namespace MR
{

/// History action for visualizeMaskType change
/// \ingroup HistoryGroup
class ChangeVisualizePropertyAction : public HistoryAction
{
public:
    using Obj = VisualObject;
    
    /// use this constructor to remember object's visualize property mask before making any changes in it
    ChangeVisualizePropertyAction( const std::string& name, const std::shared_ptr<VisualObject>& obj, AnyVisualizeMaskEnum visualizeMaskType ) :
        obj_{ obj },
        maskType_{ visualizeMaskType },
        name_{ name }
    {
        if ( obj )
            visualMask_ = obj_->getVisualizePropertyMask( maskType_ );
    }

    /// use this constructor to remember object's visualize property mask and immediately set new value
    ChangeVisualizePropertyAction( const std::string& name, const std::shared_ptr<VisualObject>& obj, AnyVisualizeMaskEnum visualizeMaskType, ViewportMask newMask ) :
        obj_{ obj },
        maskType_{ visualizeMaskType },
        name_{ name }
    {
        if ( obj )
        {
            visualMask_ = obj_->getVisualizePropertyMask( maskType_ );
            obj_->setVisualizePropertyMask( maskType_, newMask );
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        auto bufMask = obj_->getVisualizePropertyMask( maskType_ );
        obj_->setVisualizePropertyMask( maskType_, visualMask_ );
        visualMask_ = bufMask;
    }

    static void setObjectDirty( const std::shared_ptr<VisualObject>& ){ }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::shared_ptr<VisualObject> obj_;
    ViewportMask visualMask_;
    AnyVisualizeMaskEnum maskType_;
    std::string name_;
};

/// History action for object selected state
/// \ingroup HistoryGroup
class ChangeObjectSelectedAction : public HistoryAction
{
public:
    using Obj = Object;
    /// Constructed from original obj
    ChangeObjectSelectedAction( const std::string& name, const std::shared_ptr<Object>& obj ) :
        obj_{ obj },
        name_{ name }
    {
        if ( obj )
            selected_ = obj_->isSelected();
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        auto bufSel = obj_->isSelected();
        obj_->select( selected_ );
        selected_ = bufSel;
    }

    static void setObjectDirty( const std::shared_ptr<Object>& )
    {}

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::shared_ptr<Object> obj_;
    bool selected_{ false };
    std::string name_;
};

/// History action for object visibility
/// \ingroup HistoryGroup
class ChangeObjectVisibilityAction : public HistoryAction
{
public:
    using Obj = Object;
    /// Constructed from original obj
    ChangeObjectVisibilityAction( const std::string& name, const std::shared_ptr<Object>& obj ) :
        obj_{ obj },
        name_{ name }
    {
        if ( obj )
            visibilityMask_ = obj_->visibilityMask();
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        auto bufVisMask = obj_->visibilityMask();
        obj_->setVisibilityMask( visibilityMask_ );
        visibilityMask_ = bufVisMask;
    }

    static void setObjectDirty( const std::shared_ptr<Object>& )
    {}

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::shared_ptr<Object> obj_;
    ViewportMask visibilityMask_;
    std::string name_;
};

}
