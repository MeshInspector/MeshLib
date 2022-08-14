#pragma once

#include "MRObjectLinesHolder.h"

namespace MR
{

/// This object type has not visual representation, just holder for lines in scene
/// \ingroup DataModelGroup
class MRMESH_CLASS ObjectLines : public ObjectLinesHolder
{
public:
    ObjectLines() = default;
    ObjectLines( ObjectLines&& ) = default;
    ObjectLines& operator=( ObjectLines&& ) = default;
    virtual ~ObjectLines() = default;

    constexpr static const char* TypeName() noexcept { return "ObjectLines"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setPolyline( const std::shared_ptr<Polyline3>& polyline );
    /// sets given polyline to this, and returns back previous polyline of this;
    MRMESH_API virtual std::shared_ptr< Polyline3 > updatePolyline( std::shared_ptr< Polyline3 > polyline );

    virtual const std::shared_ptr<Polyline3>& varPolyline() { return polyline_; }

    MRMESH_API virtual void setDirtyFlags( uint32_t mask ) override;

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectLines( ProtectedStruct, const ObjectLines& obj ) : ObjectLines( obj ) {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;
    virtual std::string getClassName() const override { return "Lines"; }

protected:
    MRMESH_API ObjectLines( const ObjectLines& other );

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;
};

}
