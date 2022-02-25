#pragma once
#include "MRMeshFwd.h"
#include "MRObjectMesh.h"
#include "MRDistanceMapParams.h"


namespace MR
{

// This class stores information about distance map object
class MRMESH_CLASS ObjectDistanceMap : public ObjectMesh
{
public:
    MRMESH_API ObjectDistanceMap();
    ObjectDistanceMap( ObjectDistanceMap&& ) noexcept = default;
    ObjectDistanceMap& operator = ( ObjectDistanceMap&& ) noexcept = default;
    virtual ~ObjectDistanceMap() = default;

    // this ctor is public only for std::make_shared used inside clone()
    ObjectDistanceMap( ProtectedStruct, const ObjectDistanceMap& obj ) : ObjectDistanceMap( obj ) {}

    constexpr static const char* TypeName() noexcept { return "ObjectDistanceMap"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    //setters
    MRMESH_API void setDistanceMap( const std::shared_ptr<DistanceMap>& dmap, const DistanceMapToWorld& toWorldParams );
    
    //getters
    MRMESH_API const std::shared_ptr<DistanceMap>& getDistanceMap() const;

    MRMESH_API const DistanceMapToWorld& getToWorldParameters() const;

protected:

    MRMESH_API ObjectDistanceMap( const ObjectDistanceMap& other );

    // swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API tl::expected<void, std::string> deserializeModel_( const std::filesystem::path& path ) override;

    MRMESH_API virtual tl::expected<std::future<void>, std::string> serializeModel_( const std::filesystem::path& path ) const override;

private:
    std::shared_ptr<DistanceMap> dmap_;
    DistanceMapToWorld toWorldParams_;

    //rebuild mesh according sets DistanceMap & DistanceMapToWorld
    void construct_();

    // this is private function to set default colors of this type (ObjectDistanceMap) in constructor only
    void setDefaultColors_();
};

} // namespace MR
