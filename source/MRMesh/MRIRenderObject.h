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

/// describes basic rendering parameters in a viewport
struct BaseRenderParams
{
    const Matrix4f& viewMatrix;
    const Matrix4f& projMatrix;
    ViewportId viewportId;       // id of the viewport
    Vector4i viewport;           // viewport x0, y0, width, height
};

/// describes parameters necessary to render an object
struct ModelRenderParams : BaseRenderParams
{
    const Matrix4f& modelMatrix;
    const Matrix4f* normMatrixPtr{ nullptr }; // normal matrix, only necessary for triangles rendering
    const Plane3f& clipPlane;    // viewport clip plane (it is not applied while object does not have clipping flag set)
    DepthFuncion depthFunction = DepthFuncion::Default;
    Vector3f lightPos;           // position of light source, unused for picker
    bool alphaSort{ false };     // if this flag is true shader for alpha sorting is used, unused for picker
};

class IRenderObject
{
public:
    virtual ~IRenderObject() = default;
    // These functions do:
    // 1) bind data
    // 2) pass shaders arguments
    // 3) draw data
    virtual void render( const ModelRenderParams& params ) = 0;
    virtual void renderPicker( const ModelRenderParams& params, unsigned geomId ) = 0;
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