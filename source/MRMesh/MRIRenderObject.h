#pragma once
#include "MRMesh/MRFlagOperators.h"
#include "MRMeshFwd.h"
#include "MRViewportId.h"
#include "MRVector2.h"
#include "MRVector4.h"
#include "MRAffineXf3.h"
#include <functional>
#include <typeindex>
#include <memory>

namespace MR
{

class Viewport;

enum class DepthFunction
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
MR_MAKE_FLAG_OPERATORS( DepthFunction )

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
    DepthFunction depthFunction = DepthFunction::Default;
    Vector3f lightPos;           // position of light source, unused for picker
    bool alphaSort{ false };     // if this flag is true shader for alpha sorting is used, unused for picker
};

struct UiRenderParams
{
    /// Multiply all your sizes by this amount. Unless they are already premultipled, e.g. come from `ImGui::GetStyle()`.
    float scale = 1;

    /// The current viewport.
    const Viewport* viewport = nullptr;

    /// For convenience, the viewport corner coordinates (with Y going down).
    Vector2f viewportCornerA;
    Vector2f viewportCornerB;
};

struct BasicUiRenderTask
{
    virtual ~BasicUiRenderTask() = default;

    BasicUiRenderTask() = default;
    BasicUiRenderTask( const BasicUiRenderTask& ) = delete;
    BasicUiRenderTask& operator=( const BasicUiRenderTask& ) = delete;

    /// The tasks are sorted by this depth.
    float renderTaskDepth = 0;

    /// This is an optional early pass, where you can claim exclusive control over the mouse.
    /// If you want to handle clicks or hovers, do it here, only if the argument is false. Then set it to true, if you handled the click/hover.
    virtual void earlyBackwardPass( bool& mouseHoverConsumed ) { (void)mouseHoverConsumed; }

    /// This is the main rendering pass.
    virtual void renderPass() = 0;
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
    virtual size_t heapBytes() const { return 0; }
    /// returns the amount of memory this object allocated in OpenGL
    virtual size_t glBytes() const { return 0; }
    /// binds all data for this render object, not to bind ever again (until object becomes dirty)
    virtual void forceBindAll() {}

    using UiTaskList = std::vector<std::shared_ptr<BasicUiRenderTask>>;

    /// Render the ImGui UI. This is repeated for each viewport.
    /// Here you're supposed to only insert tasks into `tasks`, instead of rendering things directly.
    /// * `params` will remain alive as long as the tasks are used.
    /// * You'll have at most one living task at a time, so you can write a non-owning pointer to an internal task.
    virtual void renderUi( const UiRenderParams& params, UiTaskList& tasks ) { (void)params; (void)tasks; }
};

MRMESH_API std::unique_ptr<IRenderObject> createRenderObject( const VisualObject& visObj, const std::type_index& type );

template<typename ObjectType>
std::unique_ptr<IRenderObject> createRenderObject( const VisualObject& visObj )
{
    static_assert( std::is_base_of_v<VisualObject, std::remove_reference_t<ObjectType>>, "MR::VisualObject is not base of ObjectType" );
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
