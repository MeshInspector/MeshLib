#pragma once
#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"
#include "MRObjectMeshHolder.h"
#include "MRProgressCallback.h"
#include "MRHistogram.h"
#include "MRVolumeIndexer.h"

namespace MR
{


// This class stores information about voxels object
class MRMESH_CLASS ObjectVoxels : public ObjectMeshHolder
{
public:
    MRMESH_API ObjectVoxels();
    ObjectVoxels& operator = ( ObjectVoxels&& ) noexcept = default;
    ObjectVoxels( ObjectVoxels&& ) noexcept = default;
    virtual ~ObjectVoxels() = default;

    constexpr static const char* TypeName() noexcept { return "ObjectVoxels"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    // Returns iso surface, empty if iso value is not set
    const std::shared_ptr<Mesh>& surface() const { return mesh_; }

    // Returns Float grid which contains voxels data, see more on openvdb::FloatGrid
    const FloatGrid& grid() const
    {
        return grid_;
    }
    // Returns dimensions of voxel objects
    const Vector3i& dimensions() const
    {
        return dimensions_;
    }
    // Returns current iso value
    float getIsoValue() const
    {
        return isoValue_;
    }
    // Returns histogram
    const Histogram& histogram() const
    {
        return histogram_;
    }

    const Vector3f& voxelSize() const
    {
        return voxelSize_;
    }

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;


    // Clears all internal data and then creates grid and calculates histogram
    MRMESH_API void construct( const SimpleVolume& volume, const ProgressCallback& cb = {} );
    // Clears all internal data and calculates histogram
    MRMESH_API void construct( const FloatGrid& grid, const Vector3f& voxelSize, const ProgressCallback& cb = {} );
    // Updates histogram, by stored grid (evals min and max values from grid)
    // rebuild iso surface if it is present
    MRMESH_API void updateHistogramAndSurface( const ProgressCallback& cb = {} );

    // Sets iso value and updates iso-surfaces if needed: 
    // Returns true if iso-value was updated, false - otherwise
    MRMESH_API virtual bool setIsoValue( float iso, const ProgressCallback& cb = {} );
    // Sets active bounds for some simplifications (max excluded)
    // active bounds is box in voxel coordinates, note that voxels under (0,0,0) and voxels over (dimensions) are empty 
    MRMESH_API virtual void setActiveBounds( const Box3i& activeBox );
    // Returns active bounds (max excluded)
    // active bounds is box in voxel coordinates, note that voxels under (0,0,0) and voxels over (dimensions) are empty 
    const Box3i& getActiveBounds() const
    {
        return activeBox_;
    }

    // VoxelId is numerical representation of voxel
    // Coordinate is {x,y,z} indices of voxels in box (base dimensions space, NOT active dimensions)
    // Point is local space coordinate of point in scene
    MRMESH_API VoxelId getVoxelIdByCoordinate( const Vector3i& coord ) const;
    MRMESH_API VoxelId getVoxelIdByPoint( const Vector3f& point ) const;
    MRMESH_API Vector3i getCoordinateByVoxelId( VoxelId id ) const;

    // Returns indexer with more options
    const VolumeIndexer& getVolumeIndexer() const { return indexer_; }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setDirtyFlags( uint32_t mask ) override;

    // this ctor is public only for std::make_shared used inside clone()
    ObjectVoxels( ProtectedStruct, const ObjectVoxels& obj ) : ObjectVoxels( obj ) {}

private:
    FloatGrid grid_;
    Vector3i dimensions_;
    float isoValue_{0.0f};
    Histogram histogram_;
    Vector3f voxelSize_;
    Box3i activeBox_;

    // Service data
    VolumeIndexer indexer_ = VolumeIndexer( dimensions_ );
    Vector3f reverseVoxelSize_;

    void updateHistogram_( float min, float max );


    // this is private function to set default colors of this type (ObjectVoxels) in constructor only
    void setDefaultColors_();

protected:
    MRMESH_API ObjectVoxels( const ObjectVoxels& other );

    // swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API tl::expected<void, std::string> deserializeModel_( const std::filesystem::path& path ) override;

    MRMESH_API virtual tl::expected<std::future<void>, std::string> serializeModel_( const std::filesystem::path& path ) const override;
};


}
#endif
