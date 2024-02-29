#pragma once

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVisualObject.h"

namespace MR::RenderWrapObject
{

namespace detail
{
    // Need this helper base class to make sure the `subobject` is initialized before the render object, otherwise we get a crash.
    template <typename ObjectType>
    struct SubobjectStorage
    {
        ObjectType subobject;
    };
}

// The first template argument of `Wrapper` can inherit from this to know the object we're wrapping.
class BasicWrapperTarget
{
protected:
    ~BasicWrapperTarget() = default;
    const VisualObject* target_ = nullptr;

public:
    void setTargetObject( const VisualObject& object ) { target_ = &object; }
};

// An `IRenderObject` that embeds a data model object and another render object in it.
// The embedded render object points to the embedded data model object.
template <typename ObjectType, typename RenderObjectType>
class Wrapper : public detail::SubobjectStorage<ObjectType>, public RenderObjectType
{
public:
    Wrapper( const VisualObject& object )
        : RenderObjectType( detail::SubobjectStorage<ObjectType>::subobject )
    {
        if constexpr ( std::derived_from<ObjectType, BasicWrapperTarget> )
            this->subobject.setTargetObject( object );
    }

    Wrapper( const Wrapper& ) = delete;
    Wrapper& operator=( const Wrapper& ) = delete;
};

// This can act as the first parameter for `Wrapper` to copy most common visual properties into this subobject from the owner object.
template <typename ObjectType>
class CopyVisualProperties : public ObjectType, public BasicWrapperTarget
{
public:
    bool isSelected() const override
    {
        return target_->isSelected();
    }

    const ViewportProperty<Color>& getFrontColorsForAllViewports( bool selected = true ) const override
    {
        return target_->getFrontColorsForAllViewports( selected );
    }

    const ViewportProperty<Color>& getBackColorsForAllViewports() const override
    {
        return target_->getBackColorsForAllViewports();
    }

    const ViewportProperty<uint8_t>& getGlobalAlphaForAllViewports() const override
    {
        return target_->getGlobalAlphaForAllViewports();
    }
};


} // namespace MR::RenderWrapObject
