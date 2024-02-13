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
    constexpr static const char* TypeName() noexcept { return "RootObject"; }
    constexpr static const char* RootName() noexcept { return "Root"; }
    virtual const char* typeName() const override { return TypeName(); }
    virtual void setAncillary( bool ) override { Object::setAncillary( false ); }
    virtual bool select( bool ) override { return Object::select( false ); }
    virtual void setName( std::string ) override { Object::setName( SceneRootObject::RootName() ); }
protected:
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
    static SceneRoot& instace_();
    SceneRoot();

    std::shared_ptr<SceneRootObject> root_;

    /// path to the recently opened scene
    std::filesystem::path scenePath_;
};
}
