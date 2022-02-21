#pragma once

#include "MRVisualObject.h"
#include "MRPointCloud.h"
#include "MRXfBasedCache.h"

namespace MR
{
struct RenderParams;

class MRMESH_CLASS ObjectPoints : public VisualObject
{
public:
    MRMESH_API ObjectPoints();
    MRMESH_API ObjectPoints( const ObjectMesh& objMesh, bool saveNormals = true );
    ObjectPoints& operator = ( ObjectPoints&& ) = default;
    ObjectPoints( ObjectPoints&& ) = default;
    virtual ~ObjectPoints() = default;

    constexpr static const char* TypeName() noexcept { return "ObjectPoints"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    // returns variable point cloud, if const point cloud is needed use `pointCloud()` instead
    virtual const std::shared_ptr<PointCloud>& varPointCloud() { return points_; }
    // returns const point cloud, if variable point cloud is needed use `varPointCloud()` instead
    const std::shared_ptr<const PointCloud>& pointCloud() const 
    { return reinterpret_cast< const std::shared_ptr<const PointCloud>& >( points_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    virtual void setPointCloud( const std::shared_ptr<PointCloud>& pointCloud ) { points_ = pointCloud; setDirtyFlags( DIRTY_ALL ); }

    MRMESH_API virtual void setDirtyFlags( uint32_t mask ) override;

    // sets size of points on screen in pixels
    MRMESH_API virtual void setPointSize( float size );
    // returns size of points on screen in pixels
    float getPointSize() const { return pointSize_; }

    // swaps this object with other
    MRMESH_API virtual void swap( Object& other ) override;

    // this ctor is public only for std::make_shared used inside clone()
    ObjectPoints( ProtectedStruct, const ObjectPoints& obj ) : ObjectPoints( obj ) {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    // returns cached bounding box of this point object in world coordinates;
    // if you need bounding box in local coordinates please call getBoundingBox()
    MRMESH_API virtual Box3f getWorldBox() const override;

protected:
    ObjectPoints( const ObjectPoints& other ) = default;

    virtual Vector<Vector3f, VertId> computeVertsNormals_() const override
    {
        if ( points_ )
            return points_->normals;
        return {};
    }

    MRMESH_API virtual Box3f computeBoundingBox_() const override;
    MRMESH_API virtual Box3f computeBoundingBoxXf_() const override;

    MRMESH_API virtual tl::expected<std::future<void>, std::string> serializeModel_( const std::filesystem::path& path ) const override;

    MRMESH_API virtual tl::expected<void, std::string> deserializeModel_( const std::filesystem::path& path ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API virtual void setupRenderObject_() const override;

private:
    std::shared_ptr<PointCloud> points_;
    mutable XfBasedCache<Box3f> worldBox_;

    // size of point in pixels
    float pointSize_{ 5.0f };

    // this is private function to set default colors of this type (ObjectPoints) in constructor only
    void setDefaultColors_();
};
}