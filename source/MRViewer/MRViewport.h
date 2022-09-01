#pragma once

#include "MRViewportGL.h"
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MRPointOnFace.h>
#include <MRMesh/MRViewportId.h>
#include <MRMesh/MRQuaternion.h>
#include <MRMesh/MRMatrix4.h>
#include <MRMesh/MRColor.h>
#include <MRMesh/MRBox.h>
#include "imgui.h"
#include <memory>
#include <functional>

using ObjAndPick = std::pair<std::shared_ptr<MR::VisualObject>, MR::PointOnFace>;
using ConstObjAndPick = std::pair<std::shared_ptr<const MR::VisualObject>, MR::PointOnFace>;

namespace MR
{

// Viewport size
using ViewportRectangle = Box2f;

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
    MRVIEWER_API void set_axes_pose(const int pixelXoffset = -100, const int pixelYoffset = -100);
    MRVIEWER_API void set_axes_size(const int axisPixSize = 80);

    // Shutdown
    MRVIEWER_API void shut();

    // ------------------- Drawing functions

    // Clear the frame buffers
    MRVIEWER_API void clear_framebuffers() const;

    // Draw everything
    // forceZbuffer - rewrite Z buffer anyway
    MRVIEWER_API void draw( const VisualObject& data, const AffineXf3f& xf, bool forceZBuffer = false, bool alphaSort = false ) const;

    // Returns visual points with corresponding colors (pair<vector<Vector3f>,vector<Vector4f>>)
    MRVIEWER_API const ViewportPointsWithColors& getPointsWithColors() const;
    // Returns visual lines segments with corresponding colors (pair<vector<LineSegm3f>,vector<SegmEndColors>>)
    [[deprecated]]
    MRVIEWER_API const ViewportLinesWithColors& getLinesWithColors() const;

    // Sets visual points with corresponding colors (pair<vector<Vector3f>,vector<Vector4f>>)
    // calls 'beforeSetPointsWithColors' lambda if it is present
    [[deprecated]]
    MRVIEWER_API void setPointsWithColors( const ViewportPointsWithColors& pointsWithColors );

    // Sets visual lines segments with corresponding colors (pair<vector<LineSegm3f>,vector<SegmEndColors>>)
    // calls 'beforeSetLinesWithColors' lambda if it is present
    [[deprecated]]
    MRVIEWER_API void setLinesWithColors( const ViewportLinesWithColors& linesWithColors );

    // Add line to draw from start_point position to fin_point position.
    [[deprecated]]
    MRVIEWER_API void  add_line( const Vector3f& start_pos, const Vector3f& fin_pos,
                                 const Color& color_start = Color::black(), const Color& color_fin = Color::black() );
    // Add lines from points. 
    [[deprecated]]
    MRVIEWER_API void  add_lines( const std::vector<Vector3f>& points, const Color& color = Color::black() );
    [[deprecated]]
    MRVIEWER_API void  add_lines( const std::vector<Vector3f>& points, const std::vector<Color>& colors );
    // Remove all lines selected for draw
    [[deprecated]]
    MRVIEWER_API void  remove_lines();
    // Add point to draw-list  as a  "pos" position. 
    [[deprecated]]
    MRVIEWER_API void  add_point( const Vector3f& pos, const Color& color = Color::black() );
    // Remove all lines selected for draw
    [[deprecated]]
    MRVIEWER_API void  remove_points();
    // Is there a need to use depth_test for preview lines. (default: false)
    [[deprecated]]
    MRVIEWER_API void setPreviewLinesDepthTest( bool on );
    // Is there a need to use depth_test for preview points. (default: false)
    [[deprecated]]
    MRVIEWER_API void setPreviewPointsDepthTest( bool on );

    [[deprecated]]
    bool getPreviewLinesDepthTest() const { return previewLinesDepthTest_; }
    [[deprecated]]
    bool getPreviewPointsDepthTest() const { return previewPointsDepthTest_; }

    // Point size in pixels
    float point_size{4.0f};
    // Line width in pixels
    float line_width{1.0f};
    // Line and points z buffer offset
    // negative offset make it closer to camera
    // note that z buffer is not linear and common values are in range [0..1]
    float linesZoffset{0.0f};
    float pointsZoffset{0.0f};

    // This lambda is called before each change of visual points
    std::function<void( const ViewportLinesWithColors& curr, const ViewportLinesWithColors& next )> beforeSetLinesWithColors{};
    // This lambda is called before each change of visual lines
    std::function<void( const ViewportPointsWithColors& curr, const ViewportPointsWithColors& next )> beforeSetPointsWithColors{};

    // This function allows to pick point in scene by GL
    // comfortable usage:
    //     auto [obj,pick] = pick_render_object();
    // pick all visible and pickable objects
    // picks objects from current mouse pose by default
    MRVIEWER_API ObjAndPick pick_render_object() const;
    // This function allows to pick point in scene by GL
    // comfortable usage:
    //     auto [obj,pick] = pick_render_object( objects );
    // pick objects from input
    MRVIEWER_API ObjAndPick pick_render_object( const std::vector<VisualObject*>& objects ) const;
    // This function allows to pick point in scene by GL
    // comfortable usage:
    //     auto [obj,pick] = pick_render_object( objects );
    // pick all visible and pickable objects
    // picks objects from custom viewport point
    MRVIEWER_API ObjAndPick pick_render_object( const Vector2f& viewportPoint ) const;
    // This function allows to pick point in scene by GL
    // comfortable usage:
    //     auto [obj,pick] = pick_render_object( objects );
    // picks objects from custom viewport point
    MRVIEWER_API ObjAndPick pick_render_object( const std::vector<VisualObject*>& objects, const Vector2f& viewportPoint ) const;
    // This function allows to pick several custom viewport space points by GL
    // returns vector of pairs [obj,pick]
    MRVIEWER_API std::vector<ObjAndPick> multiPickObjects( const std::vector<VisualObject*>& objects, const std::vector<Vector2f>& viewportPoints ) const;

    // This function finds all visible objects in given rect (max excluded) in viewport space,
    // maxRenderResolutionSide - this parameter limits render resolution to improve performance
    //                           if it is too small, little objects can be lost
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    MRVIEWER_API std::vector<std::shared_ptr<VisualObject>> findObjectsInRect( const Box2i& rect, 
                                                                               int maxRenderResolutionSide = 512 ) const;

    // This function allows to pick point in scene by GL
    // comfortable usage:
    //     const auto [obj,pick] = pick_render_object();
    // pick all visible and pickable objects
    // picks objects from current mouse pose by default
    MRVIEWER_API ConstObjAndPick const_pick_render_object() const;
    // This function allows to pick point in scene by GL
    // comfortable usage:
    //      const auto [obj,pick] = pick_render_object( objects );
    // pick objects from input
    MRVIEWER_API ConstObjAndPick const_pick_render_object( const std::vector<const VisualObject*>& objects ) const;
    // This function allows to pick several custom viewport space points by GL
    // returns vector of pairs [obj,pick]
    MRVIEWER_API std::vector<ConstObjAndPick> constMultiPickObjects( const std::vector<const VisualObject*>& objects, const std::vector<Vector2f>& viewportPoints ) const;

    // multiplies view-matrix on given transformation from the _right_;
    // so if you at the same time multiplies model transformation from the _left_ on xf.inverse()
    // and call transform_view( xf ), then the user will see exactly the same picture
    MRVIEWER_API void transform_view( const AffineXf3f & xf );

    // returns base render params for immediate draw and for internal lines and points draw
    ViewportGL::BaseRenderParams getBaseRenderParams() const { return { viewM.data(), projM.data(), toVec4<int>( viewportRect_ ) }; }

    bool getRedrawFlag() const { return needRedraw_; }
    void resetRedrawFlag() const { needRedraw_ = false; }
    // ------------------- Properties

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

        // Caches the two-norm between the min/max point of the bounding box
        float objectScale{1.0f};

        Color borderColor;

        Plane3f clippingPlane{Vector3f::plusX(), 0.0f};

        mutable AffineXf3f globalBasisAxesXf; // xf representing scale of global basis in this viewport (changes each frame)
        mutable AffineXf3f basisAxesXf; // xf representing scale and translation of basis in this viewport (changes each frame)

        enum class RotationCenterMode
        {
            Static, // scene is always rotated around its center
            DynamicStatic, // scene is rotated around picked point on object, or around center, if miss pick
            Dynamic // scene is rotated around picked point on object, or around last rotation pivot, if miss pick
        } rotationMode{ RotationCenterMode::Dynamic };

        // this flag allows viewport to be selected by user
        bool selectable{true};

        bool operator==( const Viewport::Parameters& other ) const;
    };
        
    // Starts or stop rotation
    MRVIEWER_API void setRotation( bool state );

    
    MRVIEWER_API const ViewportRectangle& getViewportRect() const;

    // finds length between near pixels on zNear plane
    MRVIEWER_API const float getPixelSize() const;

    // Sets position and size of viewport:
    // rect is given as OpenGL coordinates: (0,0) is lower left corner
    MRVIEWER_API void setViewportRect( const ViewportRectangle& rect );

private:
    // Save the OpenGL transformation matrices used for the previous rendering pass
    mutable Matrix4f viewM;
    mutable Matrix4f projM;

public:
    // returns orthonormal matrix with translation
    MRVIEWER_API AffineXf3f getUnscaledViewXf() const;
    // converts directly from the view matrix
    MRVIEWER_API AffineXf3f getViewXf() const;

    // returns pure matrices. If you are going to use it for manual points transformation, you do something wrong probably
    [[deprecated]]
    MRVIEWER_API const Matrix4f& getViewMatrix() const;
    [[deprecated]]
    MRVIEWER_API const Matrix4f& getProjMatrix() const;

    // returns Y axis of view matrix. Shows Up direction with zoom-depended length
    MRVIEWER_API Vector3f getUpDirection() const;
    // returns X axis of view matrix. Shows Right direction with zoom-depended length
    MRVIEWER_API Vector3f getRightDirection() const;
    // returns Z axis of view matrix. Shows Backward direction with zoom-depended length
    MRVIEWER_API Vector3f getBackwardDirection() const;

    // returns the line from Dnear to Dfar planes for the current pixel
    // viewport space: X [0,viewport_width], Y [0,viewport_height] - (0,0) is upper left of viewport
    // Z [0.f,1.f] - 0.f is Dnear, 1.f is Dfar
    MRVIEWER_API Line3f unprojectPixelRay( const Vector2f& viewportPoint ) const;

    // convert point(s) to camera space by applying view matrix
    MRVIEWER_API Vector3f worldToCameraSpace( const Vector3f& p ) const;
    MRVIEWER_API std::vector<Vector3f> worldToCameraSpace( const std::vector<Vector3f>& p ) const;

    // projects point(s) to clip space.
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
    void setupView() const;
    // draws viewport primitives:
    //   lines: if depth test is on
    //   points: if depth test is on
    //   rotation center
    //   global basis
    void preDraw() const;
    // draws viewport primitives:
    //   lines: if depth test is off
    //   points: if depth test is off
    //   viewport border
    //   overlay basis
    void postDraw() const;

    // fill = 0.5 parameter means that scene will 0.5 of screen
    // snapView - to snap camera angle to closest canonical quaternion
    MRVIEWER_API void fitData( float fill = 1.0f, bool snapView = true );

    // Fit mode ( types of objects for which the fit is applied )
    enum class FitMode
    {
        Visible, // fit all visible objects
        SelectedPrimitives, // fit only selected primitives
        SelectedObjects, // fit only selected objects
        CustomObjectsList // fit only given objects (need additional objects list)
    };
    struct FitDataParams
    {
        float factor{ 1.f }; // part of the screen for scene location
        // snapView - to snap camera angle to closest canonical quaternion
        // orthographic view: camera moves a bit, fit FOV by the whole width or height
        // perspective view: camera is static, fit FOV to closest border.
        bool snapView{ false };
        FitMode mode{ FitMode::Visible }; // fit mode
        std::vector<std::shared_ptr<VisualObject>> objsList; // custom objects list. used only with CustomObjectsList mode

        // need for fix Clang bug
        // some as https://stackoverflow.com/questions/43819314/default-member-initializer-needed-within-definition-of-enclosing-class-outside
        FitDataParams( float factor_ = 1.f, bool snapView_ = false, FitMode mode_ = FitMode::Visible,
            const std::vector<std::shared_ptr<VisualObject>>& objsList_ = {} ) :
            factor( factor_ ),
            snapView( snapView_ ),
            mode( mode_ ),
            objsList( objsList_ )
        {};
    };
    // call fitData and change FOV to match the screen size then
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

private:
    // initializes view matrix based on camera position
    void setupViewMatrix() const;
    // returns world space to camera space transformation
    AffineXf3f getViewXf_() const;

    // initializes proj matrix based on camera angle and viewport rectangle size
    void setupProjMatrix() const;
    // initializes proj matrix for static view objects (like corner axes)
    void setupStaticProjMatrix() const;

    // use this matrix to convert world 3d point to clip point
    // clip space: XYZ [-1.f, 1.f], X axis from left(-1.f) to right(1.f), X axis from bottom(-1.f) to top(1.f),
    // Z axis from Dnear(-1.f) to Dfar(1.f)
    Matrix4f getFullViewportMatrix() const;
    Matrix4f getFullViewportInversedMatrix() const;

    ViewportRectangle viewportRect_;

    ViewportGL viewportGL_;

    bool previewLinesDepthTest_ = false;
    bool previewPointsDepthTest_ = false;

    void draw_lines() const;
    void draw_points() const;
    void draw_border() const;
    void draw_rotation_center() const;
    void draw_clipping_plane() const;
    void draw_global_basis() const;

    // init basis axis in the corner
    void init_axes();
    // Drawing basis axes in the corner
    void draw_axes() const;
    // This matrix should be used for a static objects
    // For example, basis axes in the corner
    mutable Matrix4f staticProj;
    Vector3f relPoseBase;
    Vector3f relPoseSide;

    // basis axis params
    int pixelXoffset_{ -100 };
    int pixelYoffset_{ -100 };
    int axisPixSize_{ 80 };

    // Receives point in scene coordinates, that should appear static on a screen, while rotation
    void setRotationPivot_( const Vector3f& point );
    void updateSceneBox_();
    void rotateView_() const;

    enum class Space
    {
        World,              // (x, y, z) in world space
        CameraOrthographic, // (x, y, z) in camera space
        CameraPerspective   // (x/z, y/z, z), where (x, y, z) in camera space
    };

    // finds the bounding box of given visible objects in given space
    // @param selectedPrimitives use only selected primitives of objects in calculation
    Box3f calcBox_( const std::vector<std::shared_ptr<VisualObject>>& objs, Space space, bool selectedPrimitives = false ) const;

    // find maximum FOV angle allows to keep given visible objects inside the screen
    // @returns true if all models are inside the projection volume
    // @param selectedPrimitives use only selected primitives of objects in calculation
    // @param cameraShift should be applied to the current params_.cameraTranslation value for orthographic mode
    std::pair<float, bool> getZoomFOVtoScreen_( const std::vector<std::shared_ptr<VisualObject>>& objs,
        bool selectedPrimitives = false, Vector3f* cameraShift = nullptr ) const;

    bool rotation_{ false };
    Vector3f rotationPivot_;
    Vector3f static_point_;
    Vector2f static_viewport_point;
    float distToSceneCenter_;

    mutable bool needRedraw_{false};

    // world bounding box of scene objects visible in this viewport
    Box3f sceneBox_;

    Parameters params_;
};

} //namespace MR

