#pragma once
#include "MRHistoryAction.h"
#include "MRObjectMeshHolder.h"
#include "MRObjectLinesHolder.h"
#include <memory>
#include "MRChangeVertsColorMapAction.h"

namespace MR
{

/// History action for object color palette change
/// To use with setFrontColorsForAllViewports, setBackColorsForAllViewports, setFrontColor, setBackColor
/// \ingroup HistoryGroup
class ChangeObjectColorAction : public HistoryAction
{
public:
    using Obj = VisualObject;

    enum class Type
    {
        Unselected,
        Selected,
        Back
    };

    /// Constructed from original obj
    ChangeObjectColorAction( const std::string& name, const std::shared_ptr<VisualObject>& obj, Type type ) :
        obj_{ obj },
        type_{ type },
        name_{ name }
    {
        if ( obj_ )
            colors_ = type_ == Type::Back ? obj_->getBackColorsForAllViewports() :
                obj_->getFrontColorsForAllViewports( type_ == Type::Selected );
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        ViewportProperty<Color> colors =
            type_ == Type::Back ? obj_->getBackColorsForAllViewports() :
                obj_->getFrontColorsForAllViewports( type_ == Type::Selected );
        if ( type_ == Type::Back )
            obj_->setBackColorsForAllViewports( colors_ );
        else
            obj_->setFrontColorsForAllViewports( colors_, type_ == Type::Selected );
        colors_ = colors;
    }

    static void setObjectDirty( const std::shared_ptr<VisualObject>& )
    {
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::shared_ptr<VisualObject> obj_;
    Type type_;
    ViewportProperty<Color> colors_;
    std::string name_;
};

/// History action for faces color map change
/// \ingroup HistoryGroup
class ChangeFacesColorMapAction : public HistoryAction
{
public:
    using Obj = ObjectMeshHolder;

    /// use this constructor to remember object's face colors before making any changes in them
    ChangeFacesColorMapAction( const std::string& name, const std::shared_ptr<ObjectMeshHolder>& obj ) :
        obj_{ obj },
        name_{ name }
    {
        if ( obj )
            colorMap_ = obj->getFacesColorMap();
    }

    /// use this constructor to remember object's face colors and immediate set new value
    ChangeFacesColorMapAction( const std::string& name, const std::shared_ptr<ObjectMeshHolder>& obj, FaceColors&& newColorMap ) :
        obj_{ obj },
        name_{ name }
    {
        if ( obj_ )
        {
            colorMap_ = std::move( newColorMap );
            obj_->updateFacesColorMap( colorMap_ );
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
