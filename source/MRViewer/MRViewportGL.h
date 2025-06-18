#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRVector4.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRViewportId.h"
#include "MRMesh/MRIRenderObject.h"

#include <span>

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

    // Binds and draws viewport border
    void drawBorder( const Box2f& rect, const Color& color ) const;

    // Fills viewport with given color (clear frame buffer)
    void fillViewport( const Box2f& rect, const Color& color ) const;

    // Check that members have been initialized
    bool checkInit() const;

    // Parameters of objects picking
    struct PickParameters
    {
        std::span<VisualObject* const> renderVector;       // objects to pick
        BaseRenderParams baseRenderParams;                    // parameters for rendering pick object
        Plane3f clippingPlane;                                // viewport clip plane (it is not applied while object does not have clipping flag set)
    };
    struct BasePickResult
    {
        unsigned geomId{ unsigned( -1 ) };  // id of picked object in PickParameters::renderVector (-1 means invalid)
        unsigned primId{ unsigned( -1 ) };  // id of picked primitive (-1 means invalid)
        // Note: for point clouds, primId is a point index after discretization
    };
    // Result of object picking
    struct PickResult : BasePickResult
    {
        float zBuffer{1.0f};  // camera z coordinate of picked point (1.0f means far plane)
    };
    using PickResults = std::vector<PickResult>;
    // Find picked object, face id and z coordinate, of objects given in parameters (works for vector of picks)
    PickResults pickObjects( const PickParameters& params, const std::vector<Vector2i>& picks ) const;
    // Find unique objects in given rect (return vector of ids of objects from params)
    // if maxRenderResolutionSide less then current rect size, downscale rendering for better performance
    std::vector<unsigned> findUniqueObjectsInRect( const PickParameters& params, const Box2i& rect,
                                                   int maxRenderResolutionSide ) const;

    using BasePickResults = std::vector<BasePickResult>;
    struct ScaledPickRes
    {
        BasePickResults pickRes;
        Box2i updatedBox;
    };
    // Pick all pixels in rect
    // if maxRenderResolutionSide less then current rect size, downscale rendering for better performance
    ScaledPickRes pickObjectsInRect( const PickParameters& params, const Box2i& rect,
        int maxRenderResolutionSide ) const;

private:
    struct PickColor
    {
        unsigned color[4];
    };

    struct PickTextureFrameBuffer
    {
        void resize( const Vector2i& size );
        void del();
        void bind( bool read );
    private:
        unsigned int framebuffer_{ 0 };
        unsigned int colorTexture_{ 0 };
        unsigned int renderbuffer_{ 0 };
        Vector2i size_;
    };
    mutable PickTextureFrameBuffer pickFBO_;

    std::vector<PickColor> pickObjectsInRect_( const PickParameters& params, const Box2i& rect ) const;

    bool inited_ = false;

    GLuint add_line_colors_vbo = 0;
    GLuint add_line_vbo = 0;
    GLuint add_line_vao = 0;

    GLuint add_point_colors_vbo = 0;
    GLuint add_point_vbo = 0;
    GLuint add_point_vao = 0;

    GLuint border_line_vbo = 0;
    GLuint border_line_vao = 0;
};

}
