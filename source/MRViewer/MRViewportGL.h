#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRVector4.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRLineSegm3.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRViewportId.h"

namespace MR
{
// colors of segment ends
struct SegmEndColors
{
    Vector4f a, b;
};

inline bool operator==( const SegmEndColors& a, const SegmEndColors& b )
{
    return a.a == b.a && a.b == b.b;
}

// stores points and corresponding colors (sizes of vectors should be the same)
struct ViewportPointsWithColors
{
    std::vector<Vector3f> points;
    std::vector<Vector4f> colors;
};

inline bool operator==( const ViewportPointsWithColors& a, const ViewportPointsWithColors& b )
{
    return a.points == b.points && a.colors == b.colors;
}

// store lines and corresponding colors (sizes of vectors should be the same)
struct ViewportLinesWithColors
{
    std::vector<LineSegm3f> lines;
    std::vector<SegmEndColors> colors;
};

inline bool operator==( const ViewportLinesWithColors& a, const ViewportLinesWithColors& b )
{
    return a.lines == b.lines && a.colors == b.colors;
}

// This class holds data needed to render viewport primitives and accumulative picker via OpenGL
class ViewportGL
{
public:
    typedef unsigned int GLuint;
    typedef float GLfloat;

    ViewportGL() = default;
    // Copy operators do nothing, not to share GL data
    ViewportGL( const ViewportGL& ) {}
    ViewportGL& operator = ( const ViewportGL& ) { return *this; }

    MRVIEWER_API ViewportGL( ViewportGL&& other ) noexcept;
    ViewportGL& operator = ( ViewportGL&& ) noexcept;
    ~ViewportGL();

    // Initialize all GL buffers and arrays
    void init();
    // Free all GL data
    void free();

    struct BaseRenderParams
    {
        const float* viewMatrixPtr{nullptr}; // pointer to view matrix
        const float* projMatrixPtr{nullptr}; // pointer to projection matrix
        Vector4i viewport;          // viewport x0, y0, width, height
    };

    struct RenderParams : BaseRenderParams
    {
        bool depthTest;       // depth dest of primitive
        float zOffset{0.0f};  // offset of fragments in camera z-coords
        float cameraZoom;     // camera scale factor, needed to normalize z offset
        float width;          // width of primitive
    };

    // Binds and draws viewport additional lines
    void drawLines( const RenderParams& params ) const;
    // Binds and draws viewport additional points
    void drawPoints( const RenderParams& params ) const;
    // Binds and draws viewport border
    void drawBorder( const BaseRenderParams& params, const Color& color ) const;

    // Returns visual points with corresponding colors (pair<vector<Vector3f>,vector<Vector4f>>)
    const ViewportPointsWithColors& getPointsWithColors() const;
    // Returns visual lines segments with corresponding colors (pair<vector<LineSegm3f>,vector<SegmEndColors>>)
    const ViewportLinesWithColors& getLinesWithColors() const;

    // Sets visual points with corresponding colors (pair<vector<Vector3f>,vector<Vector4f>>)
    void setPointsWithColors( const ViewportPointsWithColors& pointsWithColors );
    // Sets visual lines segments with corresponding colors (pair<vector<LineSegm3f>,vector<SegmEndColors>>)
    void setLinesWithColors( const ViewportLinesWithColors& linesWithColors );

    // Fills viewport with given color (clear frame buffer)
    void fillViewport( const Vector4i& viewport, const Color& color ) const;

    // Parameters of objects picking
    struct PickParameters
    {
        const std::vector<VisualObject*>& renderVector;       // objects to pick
        BaseRenderParams baseRenderParams;                    // parameters for rendering pick object 
        Plane3f clippingPlane;                                // viewport clip plane (it is not applied while object does not have clipping flag set)
        ViewportId viewportId;                                // viewport id
    };
    // Result of object picking
    struct PickResult
    {
        unsigned geomId{unsigned( -1 )};  // id of picked object in PickParameters::renderVector (-1 means invalid)
        unsigned primId{unsigned( -1 )};  // id of picked primitive (-1 means invalid)
        float zBuffer{1.0f};  // camera z coordinate of picked point (1.0f means far plane)
    };
    using PickResults = std::vector<PickResult>;
    // Find picked object, face id and z coordinate, of objects given in parameters (works for vector of picks)
    PickResults pickObjects( const PickParameters& params, const std::vector<Vector2i>& picks ) const;
    // Find unique objects in given rect (return vector of ids of objects from params)
    // if maxRenderResolutionSide less then current rect size, downscale rendering for better performance
    std::vector<unsigned> findUniqueObjectsInRect( const PickParameters& params, const Box2i& rect,
                                                   int maxRenderResolutionSide ) const;

    mutable bool lines_dirty = true;
    mutable bool points_dirty = true;

private:
    struct PickColor
    {
        unsigned color[4];
    };

    std::vector<PickColor> pickObjectsInRect_( const PickParameters& params, const Box2i& rect ) const;

    bool inited_ = false;

    mutable GLuint add_line_colors_vbo = 0;
    mutable GLuint add_line_vbo = 0;
    mutable GLuint add_line_vao = 0;

    mutable GLuint add_point_colors_vbo = 0;
    mutable GLuint add_point_vbo = 0;
    mutable GLuint add_point_vao = 0;

    mutable GLuint border_line_vbo = 0;
    mutable GLuint border_line_vao = 0;

    // Additional lines and points list
    ViewportLinesWithColors previewLines_;
    ViewportPointsWithColors previewPoints_;
};

}