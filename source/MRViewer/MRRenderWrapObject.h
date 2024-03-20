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

    class BasicWrapperTargetUntyped
    {
    protected:
        ~BasicWrapperTargetUntyped() = default;
    };
}

// The first template argument of `Wrapper` can inherit from this to know the object we're wrapping.
template <std::derived_from<Object> ObjectType>
class BasicWrapperTarget : public detail::BasicWrapperTargetUntyped
{
protected:
    ~BasicWrapperTarget() = default;
public:
    const ObjectType* target_ = nullptr;
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
        if constexpr ( std::derived_from<ObjectType, detail::BasicWrapperTargetUntyped> )
            this->subobject.target_ = &dynamic_cast<decltype(*this->subobject.target_)>( object );
    }

    Wrapper( const Wrapper& ) = delete;
    Wrapper& operator=( const Wrapper& ) = delete;
};

} // namespace MR::RenderWrapObject
