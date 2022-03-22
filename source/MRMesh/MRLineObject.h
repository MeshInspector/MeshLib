#pragma once
#include "MRMeshFwd.h"
#include "MRObjectLinesHolder.h"

namespace MR
{

// Object to show plane feature
class MRMESH_CLASS LineObject : public ObjectLinesHolder
{
public:
    // Creates simple plane object
    MRMESH_API LineObject();
    // Finds best plane to approx given points
    MRMESH_API LineObject( const std::vector<Vector3f>& pointsToApprox );

    LineObject( LineObject&& ) noexcept = default;
    LineObject& operator = ( LineObject&& ) noexcept = default;
    virtual ~LineObject() = default;

    constexpr static const char* TypeName() noexcept { return "LineObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    // this ctor is public only for std::make_shared used inside clone()
    LineObject( ProtectedStruct, const LineObject& obj ) : LineObject( obj )
    {}

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    // calculates direction from xf
    MRMESH_API Vector3f getDirection() const;
    // calculates center from xf
    MRMESH_API Vector3f getCenter() const;
    // updates xf to fit given normal
    MRMESH_API void setDirection( const Vector3f& normal );
    // updates xf to fit given center
    MRMESH_API void setCenter( const Vector3f& center );
    // updates xf to scale size
    MRMESH_API void setSize( float size );
protected:
    LineObject( const LineObject& other ) = default;

    // swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    virtual tl::expected<std::future<void>, std::string> serializeModel_( const std::filesystem::path& ) const override
    { return {}; }

    virtual tl::expected<void, std::string> deserializeModel_( const std::filesystem::path& ) override
    { return {}; }

private:
    void constructPolyline_();
};

}
