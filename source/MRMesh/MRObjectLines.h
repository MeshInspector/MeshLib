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

    constexpr static const char* TypeName() noexcept { return "ObjectLines"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Polyline"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Polylines"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setPolyline( const std::shared_ptr<Polyline3>& polyline );
    /// sets given polyline to this, and returns back previous polyline of this;
    MRMESH_API virtual std::shared_ptr< Polyline3 > updatePolyline( std::shared_ptr< Polyline3 > polyline );

    virtual const std::shared_ptr<Polyline3>& varPolyline() { return polyline_; }

    MRMESH_API virtual void setDirtyFlags( uint32_t mask, bool invalidateCaches = true ) override;

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectLines( ProtectedStruct, const ObjectLines& obj ) : ObjectLines( obj ) {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    /// signal about lines changing, triggered in setDirtyFlag
    using LinesChangedSignal = Signal<void( uint32_t mask )>;
    LinesChangedSignal linesChangedSignal;

protected:
    ObjectLines( const ObjectLines& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;
    /// swaps signals, used in `swap` function to return back signals after `swapBase_`
    /// pls call Parent::swapSignals_ first when overriding this function
    MRMESH_API virtual void swapSignals_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;
};

/// constructs new ObjectLines containing the union of valid data from all input objects
[[nodiscard]] MRMESH_API std::shared_ptr<ObjectLines> merge( const std::vector<std::shared_ptr<ObjectLines>>& objsLines );

/// constructs new ObjectLines containing the region of data from input object
[[nodiscard]] MRMESH_API std::shared_ptr<ObjectLines> cloneRegion( const std::shared_ptr<ObjectLines>& objLines, const UndirectedEdgeBitSet& region );

} // namespace MR
