#pragma once

#include "MRMeshFwd.h"
#ifndef MRMESH_NO_LABEL
#include "MRVisualObject.h"
#include "MRSymbolMesh.h"

namespace MR
{

/// Renders a label using ImGui, on top of all geometry without z-buffering.
/// But those are still depth-sorted among each other.
class MRMESH_CLASS ObjectImGuiLabel : public VisualObject
{
public:
    MRMESH_API ObjectImGuiLabel();

    ObjectImGuiLabel( ObjectImGuiLabel&& ) noexcept = default;
    ObjectImGuiLabel& operator = ( ObjectImGuiLabel&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept
    {
        return "ObjectImGuiLabel";
    }
    const char* typeName() const override
    {
        return TypeName();
    }

    bool hasVisualRepresentation() const override { return true; }

    MRMESH_API std::shared_ptr<Object> clone() const override;
    MRMESH_API std::shared_ptr<Object> shallowClone() const override;

    /// \note this ctor is public only for std::make_shared used inside clone()
    MRMESH_API ObjectImGuiLabel( ProtectedStruct, const ObjectImGuiLabel& obj );

    [[nodiscard]] MRMESH_API const std::string& getLabel() const;
    MRMESH_API void setLabel( std::string value );

protected:
    ObjectImGuiLabel( const ObjectImGuiLabel& other ) = default;

    /// swaps this object with other
    MRMESH_API void swapBase_( Object& other ) override;

    MRMESH_API void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API void setupRenderObject_() const override;

private:
    std::string labelText_;
};

}
#endif
