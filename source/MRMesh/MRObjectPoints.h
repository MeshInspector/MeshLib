#pragma once
#include "MRObjectPointsHolder.h"

namespace MR
{
struct RenderParams;

/// an object that stores a points
/// \ingroup DataModelGroup
class MRMESH_CLASS ObjectPoints : public ObjectPointsHolder
{
public:
    ObjectPoints() = default;
    MRMESH_API explicit ObjectPoints( const ObjectMesh& objMesh, bool saveNormals = true );
    ObjectPoints& operator = ( ObjectPoints&& ) = default;
    ObjectPoints( ObjectPoints&& ) = default;

    constexpr static const char* TypeName() noexcept { return "ObjectPoints"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Point Cloud"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Point Clouds"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    /// returns variable point cloud, if const point cloud is needed use `pointCloud()` instead
    virtual const std::shared_ptr<PointCloud>& varPointCloud() { return points_; }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    virtual void setPointCloud( const std::shared_ptr<PointCloud>& pointCloud ) { points_ = pointCloud; setDirtyFlags( DIRTY_ALL ); }
    /// sets given point cloud to this, and returns back previous mesh of this;
    /// does not touch selection
    MRMESH_API virtual void swapPointCloud( std::shared_ptr< PointCloud >& points );

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectPoints( ProtectedStruct, const ObjectPoints& obj ) : ObjectPoints( obj ) {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    MRMESH_API virtual void setDirtyFlags( uint32_t mask, bool invalidateCaches = true ) override;

    /// signal about points or normals changing, triggered in setDirtyFlag
    using ChangedSignal = Signal<void( uint32_t mask )>;
    ChangedSignal pointsChangedSignal;
    ChangedSignal normalsChangedSignal;

protected:
    ObjectPoints( const ObjectPoints& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;
    /// swaps signals, used in `swap` function to return back signals after `swapBase_`
    /// pls call Parent::swapSignals_ first when overriding this function
    MRMESH_API virtual void swapSignals_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;
};

/// constructs new ObjectPoints containing the union of valid points from all input objects
[[nodiscard]] MRMESH_API std::shared_ptr<ObjectPoints> merge( const std::vector<std::shared_ptr<ObjectPoints>>& objsPoints );

/// constructs new ObjectPoints containing the region of data from input object
/// does not copy selection
[[nodiscard]] MRMESH_API std::shared_ptr<ObjectPoints> cloneRegion( const std::shared_ptr<ObjectPoints>& objPoints, const VertBitSet& region );

/// constructs new ObjectPoints containing the packed version of input points,
/// \param newValidVerts if given, then use them instead of valid points from pts
/// \return nullptr if the operation was cancelled
[[nodiscard]] MRMESH_API std::shared_ptr<ObjectPoints> pack( const ObjectPoints& pts, Reorder reorder, VertBitSet* newValidVerts = nullptr, const ProgressCallback & cb = {} );

} //namespace MR
