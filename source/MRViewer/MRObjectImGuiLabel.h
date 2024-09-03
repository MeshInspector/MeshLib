#pragma once

#include "exports.h"

#include "MRMesh/MRVisualObject.h"

namespace MR
{

/// Renders a label using ImGui, on top of all geometry without z-buffering.
/// But those are still depth-sorted among each other.
class MRVIEWER_CLASS ObjectImGuiLabel : public VisualObject
{
public:
    MRVIEWER_API ObjectImGuiLabel();

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

    MRVIEWER_API std::shared_ptr<Object> clone() const override;
    MRVIEWER_API std::shared_ptr<Object> shallowClone() const override;

    /// \note this ctor is public only for std::make_shared used inside clone()
    MRVIEWER_API ObjectImGuiLabel( ProtectedStruct, const ObjectImGuiLabel& obj );

    [[nodiscard]] MRVIEWER_API const std::string& getLabel() const;
    MRVIEWER_API void setLabel( std::string value );

protected:
    ObjectImGuiLabel( const ObjectImGuiLabel& other ) = default;

    /// swaps this object with other
    MRVIEWER_API void swapBase_( Object& other ) override;

    MRVIEWER_API void serializeFields_( Json::Value& root ) const override;

    MRVIEWER_API void deserializeFields_( const Json::Value& root ) override;

    MRVIEWER_API void setupRenderObject_() const override;

private:
    std::string labelText_;
};

}
