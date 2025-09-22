#pragma once
#include "MRMeshFwd.h"
#include "MRObject.h"
#include <memory>

namespace MR
{

/// Object that is parent of all scene
class MRMESH_CLASS SceneRootObject final : public Object
{
public:
    MRMESH_API SceneRootObject();

    SceneRootObject( SceneRootObject&& ) noexcept = default;

    SceneRootObject& operator = ( SceneRootObject&& ) noexcept = default;

    /// \note this ctor is public only for std::make_shared used inside clone()
    SceneRootObject( ProtectedStruct, const SceneRootObject& obj ) : SceneRootObject( obj ) {}

    constexpr static const char* TypeName() noexcept { return "RootObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    constexpr static const char* ClassName() noexcept { return "Root"; }
    virtual std::string className() const override { return ClassName(); }

    constexpr static const char* ClassNameInPlural() noexcept { return "Roots"; }
    virtual std::string classNameInPlural() const override { return ClassNameInPlural(); }

    constexpr static const char* RootName() noexcept { return "Root"; }

    virtual void setAncillary( bool ) override { Object::setAncillary( false ); }

    virtual bool select( bool ) override { return Object::select( false ); }

    virtual void setName( std::string ) override { Object::setName( SceneRootObject::RootName() ); }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;

    /// same as clone but returns correct type
    MRMESH_API std::shared_ptr<SceneRootObject> cloneRoot() const;

protected:
    SceneRootObject( const SceneRootObject& other ) = default;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;
};

/// Removes all `obj` children and attaches it to newly created `SceneRootObject`
/// note that it does not respect `obj` xf
MRMESH_API std::shared_ptr<SceneRootObject> createRootFormObject( std::shared_ptr<Object> obj );

/// Singleton to store scene root object
/// \ingroup DataModelGroup
class SceneRoot
{
public:
#ifndef MR_SCENEROOT_CONST
    MRMESH_API static SceneRootObject& get();
    MRMESH_API static std::shared_ptr<SceneRootObject>& getSharedPtr();

    MRMESH_API static void setScenePath( const std::filesystem::path& scenePath );
#endif
    MRMESH_API static const SceneRootObject& constGet();
    MRMESH_API static std::shared_ptr<const SceneRootObject> constGetSharedPtr();

    MRMESH_API static const std::filesystem::path& getScenePath();

private:
    static SceneRoot& instance_();
    SceneRoot();

    std::shared_ptr<SceneRootObject> root_;

    /// path to the recently opened scene
    std::filesystem::path scenePath_;
};
}
