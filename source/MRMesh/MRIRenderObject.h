#pragma once
#include "MRMesh/MRFlagOperators.h"
#include "MRPch/MRBindingMacros.h"
#include "MRRenderModelParameters.h"
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

/// Common rendering parameters for meshes and UI.
struct BaseRenderParams
{
    const Matrix4f& viewMatrix;
    const Matrix4f& projMatrix;
    ViewportId viewportId;       // id of the viewport
    Vector4i viewport;           // viewport x0, y0, width, height
};

/// Common rendering parameters for meshes (both for primary rendering and for the picker;
/// the picker uses this as is while the primary rendering adds more fields).
struct ModelBaseRenderParams : BaseRenderParams
{
    const Matrix4f& modelMatrix;
    const Plane3f& clipPlane;    // viewport clip plane (it is not applied while object does not have clipping flag set)
    DepthFunction depthFunction = DepthFunction::Default;
};

/// Mesh rendering parameters for primary rendering (as opposed to the picker).
struct ModelRenderParams : ModelBaseRenderParams
{
    const Matrix4f* normMatrixPtr{ nullptr }; // normal matrix, only necessary for triangles rendering
    Vector3f lightPos;            // position of light source
    bool allowAlphaSort{ false }; // if true, the object can use the alpha sorting shader if it wants to

    RenderModelPassMask passMask = RenderModelPassMask::All; // Only perform rendering if `bool( passMask & desiredPass )` is true.
};

/// `IRenderObject::renderUi()` can emit zero or more or more of those tasks. They are sorted by depth every frame.
struct BasicUiRenderTask
{
    virtual ~BasicUiRenderTask() = default;

    BasicUiRenderTask() = default;
    BasicUiRenderTask( const BasicUiRenderTask& ) = default;
    BasicUiRenderTask( BasicUiRenderTask&& ) = default;
    BasicUiRenderTask& operator=( const BasicUiRenderTask& ) = default;
    BasicUiRenderTask& operator=( BasicUiRenderTask&& ) = default;

    /// The tasks are sorted by this depth, descending (larger depth = further away).
    float renderTaskDepth = 0;

    enum class InteractionMask
    {
        mouseHover = 1 << 0,
        mouseScroll = 1 << 1,
    };
    MR_MAKE_FLAG_OPERATORS_IN_CLASS( InteractionMask )

    struct BackwardPassParams
    {
        // Which interactions should be blocked by this object.
        // This is passed along between all `renderUi()` calls in a single frame, and then the end result is used.
        mutable InteractionMask consumedInteractions{};

        // If nothing else is hovered, this returns true and writes `mouseHover` to `consumedInteractions`.
        [[nodiscard]] bool tryConsumeMouseHover() const
        {
            if ( !bool( consumedInteractions & InteractionMask::mouseHover ) )
            {
                consumedInteractions |= InteractionMask::mouseHover;
                return true;
            }
            return false;
        }
    };

    /// This is an optional early pass, where you can claim exclusive control over the mouse.
    /// This pass is executed in reverse draw order.
    virtual void earlyBackwardPass( const BackwardPassParams& params ) { (void)params; }

    /// This is the main rendering pass.
    virtual void renderPass() = 0;
};

struct UiRenderParams : BaseRenderParams
{
    /// Multiply all your hardcoded sizes by this amount.
    float scale = 1;

    using UiTaskList = std::vector<std::shared_ptr<BasicUiRenderTask>>;

    // Those are Z-sorted and then executed.
    UiTaskList* tasks = nullptr;
};

struct UiRenderManager
{
    virtual ~UiRenderManager() = default;

    // This is called before doing `IRenderObject::renderUi()` on even object in a viewport. Each viewport is rendered separately.
    virtual void preRenderViewport( ViewportId viewport ) { (void)viewport; }
    // This is called after doing `IRenderObject::renderUi()` on even object in a viewport. Each viewport is rendered separately.
    virtual void postRenderViewport( ViewportId viewport ) { (void)viewport; }

    // Returns the parameters for the `IRenderObject::earlyBackwardPass()`.
    // This will be called exactly once per viewport, each time the UI in it is rendered.
    virtual BasicUiRenderTask::BackwardPassParams beginBackwardPass( ViewportId viewport, UiRenderParams::UiTaskList& tasks ) { (void)viewport; (void)tasks; return {}; }
    // After the backward pass is performed, the parameters should be passed back into this function.
    virtual void finishBackwardPass( const BasicUiRenderTask::BackwardPassParams& params ) { (void)params; }
};

class IRenderObject
{
public:
    virtual ~IRenderObject() = default;

    // These functions do:
    // 1) bind data
    // 2) pass shaders arguments
    // 3) draw data

    // Returns true if something was rendered, or false if nothing to render.
    virtual bool render( const ModelRenderParams& params ) = 0;
    virtual void renderPicker( const ModelBaseRenderParams& params, unsigned geomId ) = 0;
    /// returns the amount of memory this object occupies on heap
    virtual size_t heapBytes() const = 0;
    /// returns the amount of memory this object allocated in OpenGL
    virtual size_t glBytes() const = 0;
    /// binds all data for this render object, not to bind ever again (until object becomes dirty)
    virtual void forceBindAll() {}

    /// Render the UI. This is repeated for each viewport.
    /// Here you can either render immediately, or insert a task into `params.tasks`, which get Z-sorted.
    /// * `params` will remain alive as long as the tasks are used.
    /// * You'll have at most one living task at a time, so you can write a non-owning pointer to an internal task.
    virtual void renderUi( const UiRenderParams& params ) { (void)params; }
};
// Those dummy definitions remove undefined references in `RenderObjectCombinator` when it calls non-overridden pure virtual methods.
// We could check in `RenderObjectCombinator` if they're overridden or not, but it's easier to just define them.
inline bool IRenderObject::render( const ModelRenderParams& ) { return false; }
inline void IRenderObject::renderPicker( const ModelBaseRenderParams&, unsigned ) {}
inline size_t IRenderObject::heapBytes() const { return 0; }
inline size_t IRenderObject::glBytes() const { return 0; }

// Combines several different `IRenderObject`s into one in a meaningful way.
template <typename ...Bases>
requires ( ( std::derived_from<Bases, IRenderObject> && !std::same_as<Bases, IRenderObject> ) && ... )
class RenderObjectCombinator : public Bases...
{
public:
    RenderObjectCombinator( const VisualObject& object )
        : Bases( object )...
    {}

    bool render( const ModelRenderParams& params ) override
    {
        bool ret = false;
        // Clang 11 chokes on this if I fold from the right instead of from the left. But why?
        (void)( ..., ( ret = Bases::render( params ) || ret ) );
        return ret;
    }
    void renderPicker( const ModelBaseRenderParams& params, unsigned geomId ) override { ( Bases::renderPicker( params, geomId ), ... ); }
    size_t heapBytes() const override { return ( std::size_t{} + ... + Bases::heapBytes() ); }
    size_t glBytes() const override { return ( std::size_t{} + ... + Bases::glBytes() ); }
    void forceBindAll() override { ( Bases::forceBindAll(), ... ); }
    void renderUi( const UiRenderParams& params ) override { ( Bases::renderUi( params ), ... ); }
};

MR_BIND_IGNORE MRMESH_API std::unique_ptr<IRenderObject> createRenderObject( const VisualObject& visObj, const std::type_index& type );

template<typename ObjectType>
MR_BIND_IGNORE std::unique_ptr<IRenderObject> createRenderObject( const VisualObject& visObj )
{
    static_assert( std::is_base_of_v<VisualObject, std::remove_reference_t<ObjectType>>, "MR::VisualObject is not base of ObjectType" );
    return createRenderObject( visObj, typeid( ObjectType ) );
}

using IRenderObjectConstructorLambda = std::function<std::unique_ptr<IRenderObject>( const VisualObject& )>;

template<typename RenderObjType>
MR_BIND_IGNORE IRenderObjectConstructorLambda makeRenderObjectConstructor()
{
    return [] ( const VisualObject& visObj ) { return std::make_unique<RenderObjType>( visObj ); };
}

class RegisterRenderObjectConstructor
{
public:
    MRMESH_API RegisterRenderObjectConstructor( const std::type_index& type, IRenderObjectConstructorLambda lambda );
    MRMESH_API ~RegisterRenderObjectConstructor();

private:
    std::type_index type_;
};

#define MR_REGISTER_RENDER_OBJECT_IMPL(objectType, .../*rendObjectType*/)\
    static MR::RegisterRenderObjectConstructor __objectRegistrator##objectType{typeid(objectType),makeRenderObjectConstructor<__VA_ARGS__>()};

}
