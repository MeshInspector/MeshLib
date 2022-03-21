#pragma once

#include "MRObjectLinesHolder.h"

namespace MR
{

// This object type has not visual representation, just holder for lines in scene
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

    virtual const std::shared_ptr<Polyline3>& varPolyline() { return polyline_; }
    const std::shared_ptr<const Polyline3>& polyline() const 
    { return reinterpret_cast< const std::shared_ptr<const Polyline3>& >( polyline_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    MRMESH_API virtual void setDirtyFlags( uint32_t mask ) override;

    // this ctor is public only for std::make_shared used inside clone()
    ObjectLines( ProtectedStruct, const ObjectLines& obj ) : ObjectLines( obj ) {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

protected:
    MRMESH_API ObjectLines( const ObjectLines& other );

    // swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;
};

}
