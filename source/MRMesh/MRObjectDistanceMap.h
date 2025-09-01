#pragma once
#include "MRMeshFwd.h"
#include "MRObjectMeshHolder.h"
#include "MRDistanceMapParams.h"


namespace MR
{

/// This class stores information about distance map object
/// \ingroup DataModelGroup
class MRMESH_CLASS ObjectDistanceMap : public ObjectMeshHolder
{
public:
    MRMESH_API ObjectDistanceMap();
    ObjectDistanceMap( ObjectDistanceMap&& ) noexcept = default;
    ObjectDistanceMap& operator = ( ObjectDistanceMap&& ) noexcept = default;
    virtual ~ObjectDistanceMap() = default;

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectDistanceMap( ProtectedStruct, const ObjectDistanceMap& obj ) : ObjectDistanceMap( obj ) {}

    constexpr static const char* TypeName() noexcept { return "ObjectDistanceMap"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Distance Map"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Distance Maps"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    /// rebuilds the mesh;
    /// if it is executed in the rendering stream then you can set the needUpdateMesh = true
    /// otherwise you should set the needUpdateMesh = false and call the function calculateMesh
    /// and after finishing in the rendering stream, call the function updateMesh
    MRMESH_API bool setDistanceMap(
        const std::shared_ptr<DistanceMap>& dmap,
        const AffineXf3f& dmap2local,
        bool needUpdateMesh = true, 
        ProgressCallback cb = {} );

    /// creates a grid for this object
    MRMESH_API std::shared_ptr<Mesh> calculateMesh( ProgressCallback cb = {} ) const;
    /// updates the grid to the current one
    MRMESH_API void updateMesh( const std::shared_ptr<Mesh>& mesh );
    
    [[nodiscard]] const std::shared_ptr<DistanceMap>& getDistanceMap() const { return dmap_; }

    [[nodiscard]] virtual bool hasModel() const override { return bool( dmap_ ); }

    /// unlike the name, actually it is the transformation from distance map in local space
    MRMESH_API const AffineXf3f& getToWorldParameters() const { return dmap2local_; }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

protected:
    ObjectDistanceMap( const ObjectDistanceMap& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API Expected<void> deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} ) override;

    MRMESH_API virtual Expected<std::future<Expected<void>>> serializeModel_( const std::filesystem::path& path ) const override;

    /// reset basic object colors to their default values from the current theme
    MRMESH_API void resetFrontColor() override;

private:
    std::shared_ptr<DistanceMap> dmap_;
    AffineXf3f dmap2local_;

    const char * saveDistanceMapFormat_{ ".raw" };

    /// rebuilds the mesh;
    /// if it is executed in the rendering stream then you can set the needUpdateMesh = true
    /// otherwise you should set the needUpdateMesh = false and call the function calculateMesh
    /// and after finishing in the rendering stream, call the function updateMesh
    bool construct_(
        const std::shared_ptr<DistanceMap>& dmap,
        const AffineXf3f& dmap2local,
        bool needUpdateMesh = true,
        ProgressCallback cb = {} );

    /// this is private function to set default colors of this type (ObjectDistanceMap) in constructor only
    void setDefaultColors_();

    /// set default scene-related properties
    void setDefaultSceneProperties_();
};

} // namespace MR
