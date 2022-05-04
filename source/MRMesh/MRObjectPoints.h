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
    MRMESH_API ObjectPoints( const ObjectMesh& objMesh, bool saveNormals = true );
    ObjectPoints& operator = ( ObjectPoints&& ) = default;
    ObjectPoints( ObjectPoints&& ) = default;
    virtual ~ObjectPoints() = default;

    constexpr static const char* TypeName() noexcept { return "ObjectPoints"; }
    virtual const char* typeName() const override { return TypeName(); }

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

protected:
    MRMESH_API ObjectPoints( const ObjectPoints& other );

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;
};
}