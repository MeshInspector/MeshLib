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

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;
    virtual std::string getClassName() const override { return "Distance Map"; }

    MRMESH_API void setDistanceMap( const std::shared_ptr<DistanceMap>& dmap, const DistanceMapToWorld& toWorldParams );
    
    [[nodiscard]] const std::shared_ptr<DistanceMap>& getDistanceMap() const { return dmap_; }

    [[nodiscard]] virtual bool hasModel() const override { return bool( dmap_ ); }

    MRMESH_API const DistanceMapToWorld& getToWorldParameters() const;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

protected:
    ObjectDistanceMap( const ObjectDistanceMap& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API VoidOrErrStr deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} ) override;

    MRMESH_API virtual Expected<std::future<VoidOrErrStr>> serializeModel_( const std::filesystem::path& path ) const override;

private:
    std::shared_ptr<DistanceMap> dmap_;
    DistanceMapToWorld toWorldParams_;

    /// rebuild mesh according sets DistanceMap & DistanceMapToWorld
    void construct_();

    /// this is private function to set default colors of this type (ObjectDistanceMap) in constructor only
    void setDefaultColors_();

    /// set default scene-related properties
    void setDefaultSceneProperties_();
};

} // namespace MR
