#pragma once
#include "MRVoxelsFwd.h"

#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRHistogram.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRVoxelsVolume.h"
#include "MRMarchingCubes.h"

namespace MR
{

/// This class stores information about voxels object
/// \ingroup DataModelGroup
class MRVOXELS_CLASS ObjectVoxels : public ObjectMeshHolder
{
public:
    MRVOXELS_API ObjectVoxels();
    ObjectVoxels& operator = ( ObjectVoxels&& ) noexcept = default;
    ObjectVoxels( ObjectVoxels&& ) noexcept = default;
    virtual ~ObjectVoxels() = default;

    constexpr static const char* TypeName() noexcept { return "ObjectVoxels"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Voxel Volume"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Voxel Volumes"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    MRVOXELS_API virtual void applyScale( float scaleFactor ) override;

    /// Returns iso surface, empty if iso value is not set
    const std::shared_ptr<Mesh>& surface() const { return data_.mesh; }

    /// Return VdbVolume
    const VdbVolume& vdbVolume() const { return vdbVolume_; };
    VdbVolume& varVdbVolume() { return vdbVolume_; }

    /// Returns Float grid which contains voxels data, see more on openvdb::FloatGrid
    const FloatGrid& grid() const { return vdbVolume_.data; }

    [[nodiscard]] virtual bool hasModel() const override { return bool( vdbVolume_.data ); }

    /// Returns dimensions of voxel objects
    const Vector3i& dimensions() const
    { return vdbVolume_.dims; }
    /// Returns current iso value
    float getIsoValue() const
    { return isoValue_; }
    /// Returns histogram
    const Histogram& histogram() const
    { return histogram_; }

    const Vector3f& voxelSize() const
    { return vdbVolume_.voxelSize; }

    MRVOXELS_API virtual std::vector<std::string> getInfoLines() const override;

    /// Clears all internal data and then creates grid and calculates histogram (surface is not built, call \ref updateHistogramAndSurface)
    /// \param normalPlusGrad true means that iso-surface normals will be along gradient, false means opposite direction
    /// \param minmax optional data about known min and max values
    /// set a new background for the VdbVolume, if normalPlusGrad = true, use the maximum value, otherwise the minimum value
    MRVOXELS_API void construct( const SimpleVolume& simpleVolume, const std::optional<Vector2f> & minmax = {}, ProgressCallback cb = {}, bool normalPlusGrad = false );

    /// Clears all internal data and then creates grid and calculates histogram (surface is not built, call \ref updateHistogramAndSurface)
    /// \param normalPlusGrad true means that iso-surface normals will be along gradient, false means opposite direction
    /// set a new background for the VdbVolume, if normalPlusGrad = true, use the maximum value, otherwise the minimum value
    MRVOXELS_API void construct( const SimpleVolumeMinMax& simpleVolumeMinMax, ProgressCallback cb = {}, bool normalPlusGrad = false );

    /// Clears all internal data and then remembers grid and calculates histogram (surface is not built, call \ref updateHistogramAndSurface)
    /// \param minmax optional data about known min and max values
    MRVOXELS_API void construct( const FloatGrid& grid, const Vector3f& voxelSize, const std::optional<Vector2f> & minmax = {} );

    /// Clears all internal data and then creates grid and calculates histogram (surface is not built, call \ref updateHistogramAndSurface)
    MRVOXELS_API void construct( const VdbVolume& vdbVolume );

    /// Updates histogram, by stored grid (evals min and max values from grid)
    /// rebuild iso surface if it is present
    MRVOXELS_API void updateHistogramAndSurface( ProgressCallback cb = {} );

    /// Sets iso value and updates iso-surfaces if needed: 
    /// Returns true if iso-value was updated, false - otherwise
    MRVOXELS_API virtual Expected<bool> setIsoValue( float iso, ProgressCallback cb = {}, bool updateSurface = true );

    /// Sets external surface mesh for this object
    /// and returns back previous mesh of this
    MRVOXELS_API std::shared_ptr<Mesh> updateIsoSurface( std::shared_ptr<Mesh> mesh );
    /// Sets external vdb volume for this object
    /// and returns back previous vdb volume of this
    MRVOXELS_API VdbVolume updateVdbVolume( VdbVolume vdbVolume );
    /// Sets external histogram for this object
    /// and returns back previous histogram of this
    MRVOXELS_API Histogram updateHistogram( Histogram histogram );

    /// Calculates and return new mesh or error message
    MRVOXELS_API Expected<std::shared_ptr<Mesh>> recalculateIsoSurface( float iso, ProgressCallback cb = {} ) const;
    /// Same as above, but takes external volume
    MRVOXELS_API Expected<std::shared_ptr<Mesh>> recalculateIsoSurface( const VdbVolume& volume, float iso, ProgressCallback cb = {} ) const;
    /// Calculates and returns new histogram
    MRVOXELS_API Histogram recalculateHistogram( std::optional<Vector2f> minmax, ProgressCallback cb = {} ) const;
    /// returns true if the iso-surface is built using Dual Marching Cubes algorithm or false if using Standard Marching Cubes
    bool getDualMarchingCubes() const { return dualMarchingCubes_; }
    /// sets whether to use Dual Marching Cubes algorithm for visualization (true) or Standard Marching Cubes (false);
    /// \param updateSurface forces immediate update
    MRVOXELS_API virtual void setDualMarchingCubes( bool on, bool updateSurface = true, ProgressCallback cb = {} );
    /// set voxel point positioner for Marching Cubes (only for Standard Marching Cubes)
    virtual void setVoxelPointPositioner( VoxelPointPositioner positioner ) { positioner_ = positioner; }


    /// Sets active bounds for some simplifications (max excluded)
    /// active bounds is box in voxel coordinates, note that voxels under (0,0,0) and voxels over (dimensions) are empty 
    /// NOTE: don't forget to call `invalidateActiveBoundsCaches` if you call this function from progress bar thread
    MRVOXELS_API virtual void setActiveBounds( const Box3i& activeBox, ProgressCallback cb = {}, bool updateSurface = true );
    /// Returns active bounds (max excluded)
    /// active bounds is box in voxel coordinates, note that voxels under (0,0,0) and voxels over (dimensions) are empty 
    MRVOXELS_API const Box3i& getActiveBounds() const;
    /// Call this function in main thread post processing if you call setActiveBounds from progress bar thread
    MRVOXELS_API virtual void invalidateActiveBoundsCaches();

    const VoxelBitSet& getSelectedVoxels() const { return selectedVoxels_; }
    void selectVoxels( const VoxelBitSet& selectedVoxels ) { selectedVoxels_ = selectedVoxels; }
    
    /// get active (visible) voxels
    const VoxelBitSet& getVolumeRenderActiveVoxels() const { return volumeRenderActiveVoxels_; }
    /// set active (visible) voxels (using only in Volume Rendering mode)
    MRVOXELS_API void setVolumeRenderActiveVoxels( const VoxelBitSet& activeVoxels );

    /// VoxelId is numerical representation of voxel
    /// Coordinate is {x,y,z} indices of voxels in box (base dimensions space, NOT active dimensions)
    /// Point is local space coordinate of point in scene
    MRVOXELS_API VoxelId getVoxelIdByCoordinate( const Vector3i& coord ) const;
    MRVOXELS_API VoxelId getVoxelIdByPoint( const Vector3f& point ) const;
    MRVOXELS_API Vector3i getCoordinateByVoxelId( VoxelId id ) const;

    /// Returns indexer with more options
    const VolumeIndexer& getVolumeIndexer() const { return indexer_; }

    // prepare data for volume rendering
    // returns false if canceled or voxel data is empty
    MRVOXELS_API bool prepareDataForVolumeRendering( ProgressCallback cb = {} ) const;

    bool isVolumeRenderingEnabled() const { return volumeRendering_; }
    // this function should only be called from GUI thread because it changes rendering object,
    // it can take some time to prepare data, so you can prepare data with progress callback
    // by calling `prepareDataForVolumeRendering(cb)` function before calling this one
    MRVOXELS_API void enableVolumeRendering( bool on );
    // move volume rendering data to caller: basically used in RenderVolumeObject 
    [[nodiscard]] std::unique_ptr<SimpleVolume> getVolumeRenderingData() const { return std::move( volumeRenderingData_ ); }

    // struct to control volume rendering texture
    struct VolumeRenderingParams
    {
        // volume texture smoothing
        FilterType volumeFilterType{ FilterType::Linear };
        // shading model
        enum class ShadingType
        {
            None,
            ValueGradient,
            AlphaGradient
        } shadingType{ ShadingType::None };
        // coloring type
        enum class LutType
        {
            GrayShades,
            Rainbow,
            OneColor
        } lutType{ LutType::Rainbow };
        // color that is used for OneColor mode
        Color oneColor{ Color::white() };
        // minimum colored value (voxels with lower values are transparent)
        float min{ 0.0f };
        // maximum colored value (voxels with higher values are transparent)
        float max{ 0.0f };
        // step to sample each ray with
        // if <= 0 then default sampling is used
        float samplingStep{ -1.0f };
        // type of alpha function on texture
        enum class AlphaType
        {
            Constant,
            LinearIncreasing,
            LinearDecreasing
        } alphaType{ AlphaType::Constant };
        uint8_t alphaLimit{ 10 };
        bool operator==( const VolumeRenderingParams& )const = default;
    };
    const VolumeRenderingParams& getVolumeRenderingParams() const { return volumeRenderingParams_; }
    MRVOXELS_API void setVolumeRenderingParams( const VolumeRenderingParams& params );

    MRVOXELS_API virtual bool hasVisualRepresentation() const override;

    /// sets top limit on the number of vertices in the iso-surface
    MRVOXELS_API void setMaxSurfaceVertices( int maxVerts );
    /// gets top limit on the number of vertices in the iso-surface
    int getMaxSurfaceVertices() const { return maxSurfaceVertices_; }

    MRVOXELS_API virtual std::shared_ptr<Object> clone() const override;
    MRVOXELS_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRVOXELS_API virtual void setDirtyFlags( uint32_t mask, bool invalidateCaches = true ) override;

    /// returns cached information about the number of active voxels
    [[nodiscard]] MRVOXELS_API size_t activeVoxels() const;

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectVoxels( ProtectedStruct, const ObjectVoxels& obj ) : ObjectVoxels( obj ) {}

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRVOXELS_API virtual size_t heapBytes() const override;

    /// returns overriden file extension used to serialize voxels inside this object, nullptr means defaultSerializeVoxelsFormat()
    [[nodiscard]] const char * serializeFormat() const { return serializeFormat_; }

    /// overrides file extension used to serialize voxels inside this object: must start from '.',
    /// nullptr means serialize in defaultSerializeVoxelsFormat()
    MRVOXELS_API void setSerializeFormat( const char * newFormat );

    /// reset basic object colors to their default values from the current theme
    MRVOXELS_API void resetFrontColor() override;

    /// signal about Iso-surface changes (from updateIsoSurface)
    using IsoSurfaceChangedSignal = Signal<void()>;
    IsoSurfaceChangedSignal isoSurfaceChangedSignal;

    /// triggered by changes to voxels data
    using VoxelsChangedSignal = Signal<void()>;
    VoxelsChangedSignal voxelsChangedSignal;

private:
    VolumeRenderingParams volumeRenderingParams_;
    mutable UniquePtr<SimpleVolume> volumeRenderingData_;

    int maxSurfaceVertices_{ 5'000'000 };
    VdbVolume vdbVolume_;
    float isoValue_{0.0f};
    bool dualMarchingCubes_{true};
    VoxelPointPositioner positioner_ = {};
    Histogram histogram_;
    mutable std::optional<Box3i> activeBounds_;
    mutable std::optional<size_t> activeVoxels_;

    const char * serializeFormat_ = nullptr; //means defaultSerializeVoxelsFormat()

    /// Service data
    VolumeIndexer indexer_ = VolumeIndexer( vdbVolume_.dims );
    Vector3f reverseVoxelSize_;

    void updateHistogram_( float min, float max, ProgressCallback cb = {} );


    /// this is private function to set default colors of this type (ObjectVoxels) in constructor only
    void setDefaultColors_();

    /// set default scene-related properties
    void setDefaultSceneProperties_();

protected:
    VoxelBitSet selectedVoxels_;
    VoxelBitSet volumeRenderActiveVoxels_;

    ObjectVoxels( const ObjectVoxels& other ) = default;
    bool volumeRendering_{ false };

    /// swaps this object with other
    MRVOXELS_API virtual void swapBase_( Object& other ) override;
    /// swaps signals, used in `swap` function to return back signals after `swapBase_`
    /// pls call Parent::swapSignals_ first when overriding this function
    MRVOXELS_API virtual void swapSignals_( Object& other ) override;

    MRVOXELS_API virtual void serializeFields_( Json::Value& root ) const override;

    MRVOXELS_API void deserializeFields_( const Json::Value& root ) override;

    MRVOXELS_API Expected<void> deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} ) override;

    MRVOXELS_API virtual Expected<std::future<Expected<void>>> serializeModel_( const std::filesystem::path& path ) const override;
};

/// returns file extension used to serialize ObjectVoxels by default (if not overridden in specific object),
/// the string starts with '.'
[[nodiscard]] MRVOXELS_API const std::string & defaultSerializeVoxelsFormat();

/// sets file extension used to serialize serialize ObjectVoxels by default (if not overridden in specific object),
/// the string must start from '.'
MRVOXELS_API void setDefaultSerializeVoxelsFormat( std::string newFormat );

} //namespace MR
