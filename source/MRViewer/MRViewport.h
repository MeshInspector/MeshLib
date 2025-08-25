#pragma once

#include "MRViewportGL.h"
#include "MRFitData.h"
#include "MRMesh/MRIRenderObject.h"
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MRPointOnObject.h>
#include <MRMesh/MRViewportId.h>
#include <MRMesh/MRQuaternion.h>
#include <MRMesh/MRMatrix4.h>
#include <MRMesh/MRColor.h>
#include <MRMesh/MRBox.h>
#include "MRImGui.h"
#include <memory>
#include <functional>
#include <unordered_map>
#include <optional>
#include <span>

namespace MR
{

inline ImVec2 position( const ViewportRectangle& rect )
{
    return { rect.min.x, rect.min.y };
}

inline ImVec2 size( const ViewportRectangle& rect )
{
    return { width( rect ), height( rect ) };
}

// Conversion to Vector4 type, where:
// x, y: min.x and min.y
// z, w: width and height
template<typename T>
inline Vector4<T> toVec4( const ViewportRectangle& rect )
{
    return Vector4<T>{T( rect.min.x ), T( rect.min.y ), T( MR::width( rect ) ), T( MR::height( rect ) )};
}

// Viewport is a rectangular area, in which the objects of interest are going to be rendered.
// An application can have a number of viewports each with its own ID.
class Viewport
{
public:
    using ViewportRectangle = MR::ViewportRectangle;

    /// Return the current viewport, or the viewport corresponding to a given unique identifier from ViewerInstance()
    /// \param viewportId unique identifier corresponding to the desired viewport (current viewport if 0)
    MRVIEWER_API static Viewport& get( ViewportId viewportId = {} );

    MRVIEWER_API Viewport();
    MRVIEWER_API ~Viewport();

private: //hide copying from the user to avoid silly mistakes (copies are necessary only inside this project)
    Viewport( const Viewport & ) = default;
    Viewport & operator = ( const Viewport & ) = default;
public:
    [[nodiscard]] Viewport clone() const { return Viewport( *this ); }
    Viewport( Viewport && ) noexcept = default;
    Viewport & operator = ( Viewport && ) noexcept = default;

    // Initialization
    MRVIEWER_API void init();

    // positive pixelXoffset -> offset from the left border
    // negative pixelXoffset -> offset from the right border
    // positive pixelYoffset -> offset from the top border
    // negative pixelYoffset -> offset from the bottom border
    // axisPixSize -> length of each axes arrows in pixels
    // For example: default values -> right bottom corner with 100 offsets for the base point and 80 pixels each axis length
    MRVIEWER_API void setAxesPos( const int pixelXoffset = -100, const int pixelYoffset = -100 );
    MRVIEWER_API void setAxesSize( const int axisPixSize = 80 );

    /// returns position of basis axes in viewport space
    MRVIEWER_API const Vector2f& getAxesPosition() const;
    MRVIEWER_API float getAxesSize() const;

    /// returns projection matrix that is used for basis axes and view controller rendering
    const Matrix4f& getAxesProjectionMatrix() const { return axesProjMat_; }

    // Shutdown
    MRVIEWER_API void shut();

    // ------------------- Drawing functions

    // Clear the frame buffers
    MRVIEWER_API void clearFramebuffers();

    /// Immediate draw of given object with transformation to world taken from object's scene
    /// Returns true if something was drawn.
    MRVIEWER_API bool draw( const VisualObject& obj,
        DepthFunction depthFunc = DepthFunction::Default, RenderModelPassMask pass = RenderModelPassMask::All, bool allowAlphaSort = false ) const;

    /// Immediate draw of given object with given transformation to world
    /// Returns true if something was drawn.
    MRVIEWER_API bool draw( const VisualObject& obj, const AffineXf3f& xf,
        DepthFunction depthFunc = DepthFunction::Default, RenderModelPassMask pass = RenderModelPassMask::All, bool allowAlphaSort = false ) const;

    /// Immediate draw of given object with given transformation to world and given projection matrix
    /// Returns true if something was drawn.
    MRVIEWER_API bool draw( const VisualObject& obj, const AffineXf3f& xf, const Matrix4f & projM,
        DepthFunction depthFunc = DepthFunction::Default, RenderModelPassMask pass = RenderModelPassMask::All, bool allowAlphaSort = false ) const;

    /// Rendering parameters for immediate drawing of lines and points
    struct LinePointImmediateRenderParams : BaseRenderParams
    {
        float width{1.0f};
        bool depthTest{ true };
    };

    /// Draw lines immediately
    MRVIEWER_API void drawLines( const std::vector<LineSegm3f>& lines, const std::vector<SegmEndColors>& colors, const LinePointImmediateRenderParams & params );
    void drawLines( const std::vector<LineSegm3f>& lines, const std::vector<SegmEndColors>& colors, float width = 1, bool depthTest = true )
        { drawLines( lines, colors, { getBaseRenderParams(), width, depthTest } ); }

    /// Draw points immediately
    MRVIEWER_API void drawPoints( const std::vector<Vector3f>& points, const std::vector<Vector4f>& colors, const LinePointImmediateRenderParams & params );
    void drawPoints( const std::vector<Vector3f>& points, const std::vector<Vector4f>& colors, float width = 1, bool depthTest = true )
        { drawPoints( points, colors, { getBaseRenderParams(), width, depthTest } ); }

    struct TriCornerColors
    {
        Vector4f a, b, c;
    };

    /// Draw triangles immediately (flat shaded)
    MRVIEWER_API void drawTris( const std::vector<Triangle3f>& tris, const std::vector<TriCornerColors>& colors, const ModelRenderParams& params, bool depthTest = true );
    MRVIEWER_API void drawTris( const std::vector<Triangle3f>& tris, const std::vector<TriCornerColors>& colors, const Matrix4f& modelM = {}, bool depthTest = true );

    /// Prepares base rendering parameters for this viewport
    [[nodiscard]] BaseRenderParams getBaseRenderParams() const { return getBaseRenderParams( projM_ ); }

    /// Prepares base rendering parameters for this viewport with custom projection matrix
    [[nodiscard]] BaseRenderParams getBaseRenderParams( const Matrix4f & projM ) const
        { return { viewM_, projM, id, toVec4<int>( viewportRect_ ) }; }

    /// Prepares rendering parameters to draw a model with given transformation in this viewport
    [[nodiscard]] ModelRenderParams getModelRenderParams(
         const Matrix4f & modelM, ///< model to world transformation, this matrix will be referenced in the result
         Matrix4f * normM, ///< if not null, this matrix of normals transformation will be computed and referenced in the result
         DepthFunction depthFunc = DepthFunction::Default,
         RenderModelPassMask pass = RenderModelPassMask::All,
         bool allowAlphaSort = false ///< If not null and the object is semitransparent, enable alpha-sorting.
    ) const
    { return getModelRenderParams( modelM, projM_, normM, depthFunc, pass, allowAlphaSort ); }

    /// Prepares rendering parameters to draw a model with given transformation in this viewport with custom projection matrix
    [[nodiscard]] MRVIEWER_API ModelRenderParams getModelRenderParams( const Matrix4f & modelM, const Matrix4f & projM,
         Matrix4f * normM, ///< if not null, this matrix of normals transformation will be computed and referenced in the result
         DepthFunction depthFunc = DepthFunction::Default,
         RenderModelPassMask pass = RenderModelPassMask::All,
         bool allowAlphaSort = false ///< If not null and the object is semitransparent, enable alpha-sorting.
    ) const;

    // Predicate to additionally filter objects that should be treated as pickable.
    using PickRenderObjectPredicate = std::function<bool ( const VisualObject*, ViewportMask )>;
    // Point picking parameters.
    struct PickRenderObjectParams
    {
        // If specified, this is the target screen point. Otherwise use the mouse pos in viewport coordinates.
        std::optional<Vector2f> point;

        // Predicate to additionally filter objects that should be treated as pickable.
        PickRenderObjectPredicate predicate;

        // Radius (in pixels) of a picking area.
        // <0 defaults to `getViewerInstance().glPickRadius`.
        int pickRadius = -1;
        // Usually, from several objects that fall into the peak, the closest one along the ray is selected. However,
        // if exactPickFirst = true, then the object in which the pick exactly fell (for example, a point in point cloud)
        // will be returned as the result, even if there are others within the radius, including closer objects.
        bool exactPickFirst = true;

        // if not nullptr it can override render params for picker
        const BaseRenderParams* baseRenderParams{ nullptr };

        // This will always return `{}`. We need the functions because `= {}`
        //   can't be used directly inside default arguments in the same class.
        // You don't have to use this function.
        static PickRenderObjectParams defaults()
        {
            return {};
        }
    };
    // This function allows to pick point in scene by GL with given parameters.
    // This overload uses all objects in the scene (possibly filtered by a predicate).
    MRVIEWER_API ObjAndPick pickRenderObject( const PickRenderObjectParams& params = PickRenderObjectParams::defaults() ) const;
    // This overload uses objects from the list (possibly filtered by a predicate).
    MRVIEWER_API ObjAndPick pickRenderObject( std::span<VisualObject* const> objects, const PickRenderObjectParams& params = PickRenderObjectParams::defaults() ) const;
    // This overload uses const objects from the list (possibly filtered by a predicate). Sadly need a different name to avoid over resolution issues.
    MRVIEWER_API ConstObjAndPick pickRenderObjectConst( std::span<const VisualObject* const> objects, const PickRenderObjectParams& params = PickRenderObjectParams::defaults() ) const;

    // This function allows to pick point in scene by GL
    // use default pick radius
    // comfortable usage:
    //     auto [obj,pick] = pick_render_object();
    // pick all visible and pickable objects
    // picks objects from current mouse pose by default
    // [[deprecated("Use `pickRenderObject()`")]] // Should eventually deprecate this?
    MRVIEWER_API ObjAndPick pick_render_object() const;
    [[deprecated("Use `pickRenderObject()`")]]
    MRVIEWER_API ObjAndPick pick_render_object( uint16_t pickRadius ) const;
    // This function allows to pick point in scene by GL
    // use default pick radius
    // comfortable usage:
    //     auto [obj,pick] = pick_render_object( objects );
    // pick objects from input
    [[deprecated("Use `pickRenderObject( objects } )`")]] // NOTE! If your list is hardcoded, use `.objects = std::array{ a, b, c }`.
    MRVIEWER_API ObjAndPick pick_render_object( const std::vector<VisualObject*>& objects ) const;
    // This function allows to pick point in scene by GL with a given peak radius.
    // usually, from several objects that fall into the peak, the closest one along the ray is selected.However
    // if exactPickFirst = true, then the object in which the pick exactly fell( for example, a point in point cloud )
    // will be returned as the result, even if there are others within the radius, including closer objects.
    [[deprecated("Use `pickRenderObject( ... )`")]]
    MRVIEWER_API ObjAndPick pick_render_object( const std::vector<VisualObject*>& objects, uint16_t pickRadius, bool exactPickFirst = true ) const;
    // This function allows to pick point in scene by GL with default pick radius, but with specified exactPickFirst parameter (see description upper).
    [[deprecated("Use `pickRenderObject( { .exactPickFirst = ... } )`")]]
    MRVIEWER_API ObjAndPick pick_render_object( bool exactPickFirst ) const;
    // This function allows to pick point in scene by GL
    // comfortable usage:
    //     auto [obj,pick] = pick_render_object( objects );
    // pick all visible and pickable objects
    // picks objects from custom viewport point
    [[deprecated("Use `pickRenderObject( { .point = ... } )`")]]
    MRVIEWER_API ObjAndPick pick_render_object( const Vector2f& viewportPoint ) const;
    // This function allows to pick point in scene by GL
    // comfortable usage:
    //     auto [obj,pick] = pick_render_object( objects );
    // picks objects from custom viewport point
    [[deprecated("Use `multiPickObjects( objects, { .point = ... } )`")]]
    MRVIEWER_API ObjAndPick pick_render_object( const std::vector<VisualObject*>& objects, const Vector2f& viewportPoint ) const;

    // This function allows to pick several custom viewport space points by GL
    // returns vector of pairs [obj,pick]
    // To hardcode the list of `objects`, use `{{ a, b, c }}`.
    MRVIEWER_API std::vector<ObjAndPick> multiPickObjects( std::span<VisualObject* const> objects, const std::vector<Vector2f>& viewportPoints, const BaseRenderParams* overrideRenderParams = nullptr ) const;

    // This function finds all visible objects in given rect (max excluded) in viewport space,
    // maxRenderResolutionSide - this parameter limits render resolution to improve performance
    //                           if it is too small, little objects can be lost
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    MRVIEWER_API std::vector<std::shared_ptr<VisualObject>> findObjectsInRect( const Box2i& rect,
                                                                               int maxRenderResolutionSide = 512 ) const;

    // This functions finds all visible faces in given includePixBs in viewport space,
    // maxRenderResolutionSide - this parameter limits render resolution to improve performance
    //                           if it is too small, little faces can be lost
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    MRVIEWER_API std::unordered_map<std::shared_ptr<ObjectMesh>, FaceBitSet> findVisibleFaces( const BitSet& includePixBs,
        int maxRenderResolutionSide = 512 ) const;

    /// finds all triangles of a mesh that having normals oriented toward the camera in this viewport
    [[nodiscard]] MRVIEWER_API FaceBitSet findCameraLookingFaces( const Mesh& mesh, const AffineXf3f& meshToWorld ) const;


    // This function allows to pick point in scene by GL
    // comfortable usage:
    //     const auto [obj,pick] = pick_render_object();
    // pick all visible and pickable objects
    // picks objects from current mouse pose by default
    [[deprecated("Use `pickRenderObject()`.")]]
    MRVIEWER_API ConstObjAndPick const_pick_render_object() const;
    // This function allows to pick point in scene by GL
    // comfortable usage:
    //      const auto [obj,pick] = pick_render_object( objects );
    // pick objects from input
    [[deprecated("Use `pickRenderObject( objects )`.")]]
    MRVIEWER_API ConstObjAndPick const_pick_render_object( const std::vector<const VisualObject*>& objects ) const;
    // This function allows to pick several custom viewport space points by GL
    // returns vector of pairs [obj,pick]
    MRVIEWER_API std::vector<ConstObjAndPick> constMultiPickObjects( const std::vector<const VisualObject*>& objects, const std::vector<Vector2f>& viewportPoints ) const;

    // multiplies view-matrix on given transformation from the _right_;
    // so if you at the same time multiplies model transformation from the _left_ on xf.inverse()
    // and call transformView( xf ), then the user will see exactly the same picture
    MRVIEWER_API void transformView( const AffineXf3f & xf );

    bool getRedrawFlag() const { return needRedraw_; }
    void resetRedrawFlag() { needRedraw_ = false; }

    // Unique identifier
    ViewportId id{ 1};

    struct Parameters
    {
        Color backgroundColor = Color( Vector3f{0.3f, 0.3f, 0.5f} );
        Vector3f lightPosition{0.0f, 0.3f, 0.0f};// x y z - position, w - factor

        Quaternionf cameraTrackballAngle;
        Vector3f cameraTranslation;
        float cameraZoom{1.0f};
        float cameraViewAngle{45.0f};
        float cameraDnear{1.0f};
        float cameraDfar{100.0f};

        bool depthTest{true};
        bool orthographic{true};

        enum class GlobalBasisScaleMode
        {
            Auto, // uses current scene size
            Fixed // uses global basis object internal size (one can change it with globalBasisAxes->setXf( AffineXf3f::linear( Matrix3f::scale( size ) ) ) )
        } globalBasisScaleMode{ GlobalBasisScaleMode::Auto };

        // Caches the two-norm between the min/max point of the bounding box
        float objectScale{1.0f};

        Color borderColor;

        std::string label;

        Plane3f clippingPlane{Vector3f::plusX(), 0.0f};

        enum class RotationCenterMode
        {
            Static, // scene is always rotated around its center or another manually set point
            DynamicStatic, // scene is rotated around picked point on object, or around center, if miss pick
            Dynamic // scene is rotated around picked point on object, or around last rotation pivot, if miss pick
        } rotationMode{ RotationCenterMode::Dynamic };

        // in Static RotationMode, this world space point is used as rotation pivot
        // if it is not set, scene box center is used instead
        std::optional<Vector3f> staticRotationPivot;

        // if it is true, while rotation is enabled camera can be moved along forward axis
        // in order to keep constant distance to scene center
        bool compensateRotation{ true };

        // this flag allows viewport to be selected by user
        bool selectable{true};

        bool operator==( const Viewport::Parameters& other ) const = default;
    };

    // Starts or stop rotation
    MRVIEWER_API void setRotation( bool state );

    // Note, Y is up for this box.
    MRVIEWER_API const ViewportRectangle& getViewportRect() const;

    // Finds length between near pixels on zNear plane. Only good in the orthographic projection.
    MRVIEWER_API float getPixelSize() const;

    // Finds the pixel scale at a specific world point. This works in both perspective and orthographic projection.
    // The UI scale is NOT baked into this. You have to multiply by the scale manually if you need that.
    MRVIEWER_API float getPixelSizeAtPoint( const Vector3f& worldPoint ) const;

    // Sets position and size of viewport:
    // rect is given as OpenGL coordinates: (0,0) is lower left corner
    MRVIEWER_API void setViewportRect( const ViewportRectangle& rect );

private:
    // Save the OpenGL transformation matrices used for the previous rendering pass
    Matrix4f viewM_;
    Matrix4f projM_;

public:
    /// returns orthonormal matrix with translation
    [[nodiscard]] MRVIEWER_API AffineXf3f getUnscaledViewXf() const;

    /// converts directly from the view matrix
    [[nodiscard]] AffineXf3f getViewXf() const { return AffineXf3f( viewM_ ); }

    /// returns unit vector in world space corresponding to up-direction in camera space
    [[nodiscard]] Vector3f getUpDirection() const { return Vector3f( viewM_.y.x, viewM_.y.y, viewM_.y.z ).normalized(); } // assume that viewM is orthogonal and inverse=transpose

    /// returns unit vector in world space corresponding to right-direction in camera space
    [[nodiscard]] Vector3f getRightDirection() const { return Vector3f( viewM_.x.x, viewM_.x.y, viewM_.x.z ).normalized(); } // assume that viewM is orthogonal and inverse=transpose

    /// returns unit vector in world space corresponding to direction toward camera in camera space
    [[nodiscard]] Vector3f getBackwardDirection() const { return Vector3f( viewM_.z.x, viewM_.z.y, viewM_.z.z ).normalized(); } // assume that viewM is orthogonal and inverse=transpose

    // returns the line from Dnear to Dfar planes for the current pixel
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    // Z [0.f,1.f] - 0.f is Dnear, 1.f is Dfar
    MRVIEWER_API Line3f unprojectPixelRay( const Vector2f& viewportPoint ) const;

    // convert point(s) to camera space by applying view matrix
    MRVIEWER_API Vector3f worldToCameraSpace( const Vector3f& p ) const;
    MRVIEWER_API std::vector<Vector3f> worldToCameraSpace( const std::vector<Vector3f>& p ) const;

    // projects point(s) to clip space. (rather, to normalized device coordinates, as it includes perspective division)
    // clip space: XYZ [-1.f, 1.f], X axis from left(-1.f) to right(1.f), Y axis from bottom(-1.f) to top(1.f),
    // Z axis from Dnear(-1.f) to Dfar(1.f)
    MRVIEWER_API Vector3f projectToClipSpace( const Vector3f& worldPoint ) const;
    MRVIEWER_API std::vector<Vector3f> projectToClipSpace( const std::vector<Vector3f>& worldPoints ) const;
    MRVIEWER_API Vector3f unprojectFromClipSpace( const Vector3f& clipPoint ) const;
    MRVIEWER_API std::vector<Vector3f> unprojectFromClipSpace( const std::vector<Vector3f>& clipPoints ) const;

    // project point and convert coordinates to viewport space
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    // Z [0.f,1.f] - 0.f is Dnear, 1.f is Dfar
    MRVIEWER_API Vector3f projectToViewportSpace( const Vector3f& worldPoint ) const;
    MRVIEWER_API std::vector<Vector3f> projectToViewportSpace( const std::vector<Vector3f>& worldPoints ) const;
    // unproject coordinates from viewport space
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    // Z [0.f,1.f] - 0.f is Dnear, 1.f is Dfar
    MRVIEWER_API Vector3f unprojectFromViewportSpace( const Vector3f& viewportPoint ) const;
    MRVIEWER_API std::vector<Vector3f> unprojectFromViewportSpace( const std::vector<Vector3f>& viewportPoints ) const;

    // conversations between clip space and viewport space
    // clip space: XYZ [-1.f, 1.f], X axis from left(-1.f) to right(1.f), Y axis from bottom(-1.f) to top(1.f),
    // Z axis from Dnear(-1.f) to Dfar(1.f)
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    // Z [0.f,1.f] - 0.f is Dnear, 1.f is Dfar
    MRVIEWER_API Vector3f clipSpaceToViewportSpace( const Vector3f& p ) const;
    MRVIEWER_API std::vector<Vector3f> clipSpaceToViewportSpace( const std::vector<Vector3f>& p ) const;
    MRVIEWER_API Vector3f viewportSpaceToClipSpace( const Vector3f& p ) const;
    MRVIEWER_API std::vector<Vector3f> viewportSpaceToClipSpace( const std::vector<Vector3f>& p ) const;

    // updates view and projection matrices due to camera parameters (called each frame)
    void setupView();
    // draws viewport primitives:
    //   lines: if depth test is on
    //   points: if depth test is on
    //   rotation center
    //   global basis
    void preDraw();
    // draws viewport primitives:
    //   lines: if depth test is off
    //   points: if depth test is off
    //   viewport border
    //   overlay basis
    void postDraw() const;

    // fit camera to the scene box (note: scene box does not include ancillary objects)
    // fill = 1.0 parameter means that scene will be approximately 0.5 of screen
    // snapView - to snap camera angle to closest canonical quaternion
    MRVIEWER_API void fitData( float fill = 1.0f, bool snapView = true );

    // set scene box by given one and fit camera to it
    // fill = 1.0 parameter means that box diagonal will be approximately 0.5 of the viewport
    // snapView - to snap camera angle to closest canonical quaternion
    MRVIEWER_API void fitBox( const Box3f& newSceneBox, float fill = 1.0f, bool snapView = true );

    using FitMode = MR::FitMode;
    using BaseFitParams = MR::BaseFitParams;
    using FitDataParams = MR::FitDataParams;
    using FitBoxParams = MR::FitBoxParams;

    // fit view and proj matrices to match the screen size with given box
    MRVIEWER_API void preciseFitBoxToScreenBorder( const FitBoxParams& params );
    // fit view and proj matrices to match the screen size with given objects
    MRVIEWER_API void preciseFitDataToScreenBorder( const FitDataParams& params = {} );

    // returns viewport width/height ratio
    MRVIEWER_API float getRatio() const;

    // returns true if all models are fully projected inside the viewport rectangle
    MRVIEWER_API bool allModelsInsideViewportRectangle() const;

    MRVIEWER_API const Box3f& getSceneBox() const;

    const Parameters& getParameters() const { return params_; }

    // returns camera world location for the current view
    MRVIEWER_API Vector3f getCameraPoint() const;
    // sets camera world location for the current view
    MRVIEWER_API void setCameraPoint( const Vector3f& cameraWorldPos );

    MRVIEWER_API void setCameraTrackballAngle( const Quaternionf& rot );

    MRVIEWER_API void setCameraTranslation( const Vector3f& translation );

    MRVIEWER_API void setCameraViewAngle( float newViewAngle );

    MRVIEWER_API void setCameraZoom( float zoom );

    MRVIEWER_API void setOrthographic( bool orthographic );

    MRVIEWER_API void setBackgroundColor( const Color& color );

    MRVIEWER_API void setClippingPlane( const Plane3f& plane );

    MRVIEWER_API void setLabel( std::string s );

    void setSelectable( bool on ) { params_.selectable = on; }

    MRVIEWER_API void showAxes( bool on );
    MRVIEWER_API void showClippingPlane( bool on );
    MRVIEWER_API void showRotationCenter( bool on );
    MRVIEWER_API void showGlobalBasis( bool on );
    MRVIEWER_API void rotationCenterMode( Parameters::RotationCenterMode mode );

    MRVIEWER_API void setParameters( const Viewport::Parameters& params );

    // Set camera look direction and up direction (they should be perpendicular)
    // this function changes camera position and do not change camera spot (0,0,0) by default
    // to change camera position use setCameraTranslation after this function
    MRVIEWER_API void cameraLookAlong( const Vector3f& dir, const Vector3f& up );

    // Rotates camera around axis +direction applied to axis point
    // note: this can make camera clip objects (as far as distance to scene center is not fixed)
    MRVIEWER_API void cameraRotateAround( const Line3f& axis, float angle );

    /// returns current rotation pivot in world space, which should appear static on a screen during rotation by the user
    Vector3f getRotationPivot() const { return rotationPivot_; }

    /// sets world point to be used as rotation pivot in static mode
    /// this should appear static on a screen during rotation by the user
    /// if nullopt - scene box center is used
    /// please note that this point should be set *before rotation starts*
    MRVIEWER_API void resetStaticRotationPivot( const std::optional<Vector3f>& pivot = std::nullopt );
private:
    /// sets current rotation pivot in world space, which should appear static on a screen during rotation by the user
    /// caller of this function should respect `params_.rotationMode` and `params_.staticRotationPivot`
    void setRotationPivot_( const Vector3f& point ) { rotationPivot_ = point; }
    // initializes view matrix based on camera position
    void setupViewMatrix_();
    // returns world space to camera space transformation
    AffineXf3f getViewXf_() const;

    // initializes proj matrix based on camera angle and viewport rectangle size
    void setupProjMatrix_();
    // initializes proj matrix for static view objects (like corner axes)
    void setupAxesProjMatrix_();

    // use this matrix to convert world 3d point to clip point
    // clip space: XYZ [-1.f, 1.f], X axis from left(-1.f) to right(1.f), X axis from bottom(-1.f) to top(1.f),
    // Z axis from Dnear(-1.f) to Dfar(1.f)
    Matrix4f getFullViewportMatrix() const { return projM_ * viewM_; }
    Matrix4f getFullViewportInversedMatrix() const;

    ViewportRectangle viewportRect_;

    ViewportGL viewportGL_;

    bool previewLinesDepthTest_ = false;
    bool previewPointsDepthTest_ = false;

    void draw_border() const;
    void draw_rotation_center() const;
    void draw_clipping_plane() const;
    void drawGlobalBasis() const;

    // init basis axis in the corner
    void initBaseAxes();
    // Drawing basis axes and view controller cube in the corner
    void drawAxesAndViewController() const;

    // This matrix should be used for a static objects
    // For example, basis axes in the corner
    Matrix4f axesProjMat_;
    Vector2f basisAxesPos_;
    float basisAxesSize_;

    // basis axis params
    int pixelXoffset_{ -100 };
    int pixelYoffset_{ -100 };
    int axisPixSize_{ 70 };

    void updateSceneBox_();
    void rotateView_();

    enum class Space
    {
        World,              // (x, y, z) in world space
        CameraOrthographic, // (x, y, z) in camera space
        CameraPerspective   // (x/z, y/z, z), where (x, y, z) in camera space
    };

    /**
     * @brief finds the bounding box of given visible objects in given space
     * @param selectedPrimitives use only selected primitives of objects in calculation
     */
    Box3f calcBox_( const std::vector<std::shared_ptr<VisualObject>>& objs, Space space, bool selectedPrimitives = false ) const;

    /**
     * @brief find minimum FOV angle allows to keep box given by getBoxFn visible inside the screen
     * The box is either in CameraOrthographic or CameraPerspective, depending on viewport setting
     * If cameraShift is not null, calculate angle assuming the camera can be moved, and fills the shift value
     * (orthogonal only; for perspective mode, must be null)
     * @returns true if all models are inside the projection volume
     */
    std::pair<float, bool> getZoomFOVtoScreen_( std::function<Box3f()> getBoxFn, Vector3f* cameraShift = nullptr ) const;
    // fit view and proj matrices to match the screen size with boxes returned by getBoxFn
    // getBoxFn( true ) - always camera space (respecting projection)
    // getBoxFn( false ) - if orthographic - camera space, otherwise - world space
    // getBoxFn/globalBasis - if true then getBoxFn should return box of global basis object (separately, not to interfere with actual scene size)
    void preciseFitToScreenBorder_( std::function<Box3f( bool zoomFOV, bool globalBasis )> getBoxFn, const BaseFitParams& params );

    bool rotation_{ false };
    Vector3f rotationPivot_;
    Vector3f static_point_;
    Vector2f static_viewport_point;
    float distToSceneCenter_;

    bool needRedraw_{false};

    // world bounding box of scene objects visible in this viewport
    Box3f sceneBox_;

    Parameters params_;
};

} //namespace MR
