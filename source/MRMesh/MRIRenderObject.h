#pragma once
#include "MRMeshFwd.h"
#include "MRViewportId.h"
#include "MRVector4.h"
#include <functional>
#include <typeindex>
#include <memory>

namespace MR
{

struct BaseRenderParams
{
    const float* viewMatrixPtr{ nullptr };  // pointer to view matrix
    const float* modelMatrixPtr{ nullptr }; // pointer to model matrix
    const float* projMatrixPtr{ nullptr };  // pointer to projection matrix
    const float* normMatrixPtr{ nullptr };  // pointer to norm matrix (this is used to simplify lighting calculations)
    ViewportId viewportId;       // id of current viewport
    const Plane3f& clipPlane;    // viewport clip plane (it is not applied while object does not have clipping flag set)
    Vector4i viewport;           // viewport x0, y0, width, height
};

struct RenderParams : BaseRenderParams
{
    const Vector3f& lightPos; // position of light source
    bool forceZBuffer{ false }; // if this flag is set, rewrite Z buffer anyway
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
    virtual void render( const RenderParams& params ) const = 0;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) const = 0;
    /// returns the amount of memory this object occupies on heap
    virtual size_t heapBytes() const = 0;
    /// the way the internal buffers are dealt with
    enum BufferMode {
        /// preserve the buffers to reduce re-allocation count
        AllocationEfficient,
        /// clear the buffers on every update to reduce memory consumption
        MemoryEfficient,
    };
    /// returns internal buffer mode
    virtual BufferMode getBufferMode() const { return AllocationEfficient; }
    /// sets internal buffer mode
    virtual void setBufferMode( BufferMode bufferMode ) {}
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