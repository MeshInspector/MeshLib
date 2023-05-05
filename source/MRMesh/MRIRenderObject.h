#pragma once
#include "MRMeshFwd.h"
#include "MRViewportId.h"
#include "MRVector4.h"
#include <functional>
#include <typeindex>
#include <memory>

namespace MR
{

enum class DepthFuncion
{
    Never = 0,
    Less = 1,
    Equal = 2,
    Greater = 4,
    LessOrEqual = Less | Equal,
    GreaterOrEqual = Greater | Equal,
    NotEqual = Less | Greater,
    Always = Less | Equal | Greater,
    Default = 8 // usually "Less" but may differ for different object types
};

struct BaseRenderParams
{
    const Matrix4f& viewMatrix;
    const Matrix4f& modelMatrix;
    const Matrix4f& projMatrix;
    const Matrix4f* normMatrixPtr{ nullptr }; // optional normal matrix
    ViewportId viewportId;       // id of current viewport
    const Plane3f& clipPlane;    // viewport clip plane (it is not applied while object does not have clipping flag set)
    Vector4i viewport;           // viewport x0, y0, width, height
    DepthFuncion depthFunction = DepthFuncion::Default;
};

struct RenderParams : BaseRenderParams
{
    const Vector3f& lightPos; // position of light source
    bool alphaSort{ false };    // if this flag is true shader for alpha sorting is used
};

class IRenderObject
{
public:
    virtual ~IRenderObject() = default;
    // These functions do:
    // 1) bind data
    // 2) pass shaders arguments
    // 3) draw data
    virtual void render( const RenderParams& params ) = 0;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) = 0;
    /// returns the amount of memory this object occupies on heap
    virtual size_t heapBytes() const = 0;
    /// returns the amount of memory this object allocated in OpenGL
    virtual size_t glBytes() const = 0;
    /// binds all data for this render object, not to bind ever again (until object becomes dirty)
    virtual void forceBindAll() {}
};

MRMESH_API std::unique_ptr<IRenderObject> createRenderObject( const VisualObject& visObj, const std::type_index& type );

template<typename ObjectType>
std::unique_ptr<IRenderObject> createRenderObject( const VisualObject& visObj )
{
    static_assert( std::is_base_of_v<VisualObject, ObjectType>, "MR::VisualObject is not base of ObjectType" );
    return createRenderObject( visObj, typeid( ObjectType ) );
}

using IRenderObjectConstructorLambda = std::function<std::unique_ptr<IRenderObject>( const VisualObject& )>;

template<typename RenderObjType>
IRenderObjectConstructorLambda makeRenderObjectConstructor()
{
    return [] ( const VisualObject& visObj ) { return std::make_unique<RenderObjType>( visObj ); };
}

class RegisterRenderObjectConstructor
{
public:
    MRMESH_API RegisterRenderObjectConstructor( const std::type_index& type, IRenderObjectConstructorLambda lambda );
};

#define MR_REGISTER_RENDER_OBJECT_IMPL(objectType,rendObjectType)\
    static MR::RegisterRenderObjectConstructor __objectRegistrator##objectType{typeid(objectType),makeRenderObjectConstructor<rendObjectType>()};

}