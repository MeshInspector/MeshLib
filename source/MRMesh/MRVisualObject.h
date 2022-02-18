#pragma once

#include "MRObject.h"
#include "MRBox.h"
#include "MRMeshTexture.h"
#include "MRVector.h"
#include "MRColor.h"
#include "MRMeshLabel.h"
#include "MRIRenderObject.h"
#include "MRMeshNormals.h"

namespace MR
{
// Type of mesh coloring,
// note that texture are applied over main coloring
enum class ColoringType
{
    SolidColor,   // Use simple color for whole mesh
    PrimitivesColorMap, // Use different color (taken from faces colormap) for each ptimitive
    FacesColorMap = PrimitivesColorMap, // Use different color (taken from faces colormap) for each face (primitive for object mesh)
    LinesColorMap = PrimitivesColorMap, // Use different color (taken from faces colormap) for each line (primitive for object lines)
    VertsColorMap  // Use different color (taken from verts colormap) for each vert
};

struct VisualizeMaskType
{
    enum Type : unsigned
    {
        Visibility,
        Texture,
        InvertedNormals,
        Name,
        Labels,
        CropLabelsByViewportRect,
        ClippedByPlane,
        DepthTest,

        VisualizePropsCount
    };
};

using AllVisualizeProperties = std::vector<ViewportMask>;

enum DirtyFlags
{
    DIRTY_NONE = 0x0000,
    DIRTY_POSITION = 0x0001,
    DIRTY_UV = 0x0002,
    DIRTY_VERTS_RENDER_NORMAL = 0x0004, // gl normals
    DIRTY_FACES_RENDER_NORMAL = 0x0008, // gl normals
    DIRTY_CORNERS_RENDER_NORMAL = 0x0010, // gl normals
    DIRTY_RENDER_NORMALS = DIRTY_VERTS_RENDER_NORMAL | DIRTY_FACES_RENDER_NORMAL | DIRTY_CORNERS_RENDER_NORMAL,
    DIRTY_SELECTION = 0x0020,
    DIRTY_TEXTURE = 0x0040,
    DIRTY_PRIMITIVES = 0x0080,
    DIRTY_FACE = DIRTY_PRIMITIVES,
    DIRTY_BACK_FACES = 0x0100,
    DIRTY_VERTS_COLORMAP = 0x0200,
    DIRTY_PRIMITIVE_COLORMAP = 0x0400,
    DIRTY_FACES_COLORMAP = DIRTY_PRIMITIVE_COLORMAP,
    DIRTY_MESH = 0x07FF,
    DIRTY_BOUNDING_BOX = 0x0800,
    DIRTY_BOUNDING_BOX_XF = 0x1000,
    DIRTY_BORDER_LINES = 0x2000,
    DIRTY_EDGES_SELECTION = 0x4000,
    DIRTY_VERTS_NORMAL = 0x8000,   // object normals
    DIRTY_FACES_NORMAL = 0x10000,   // object normals
    DIRTY_CORNERS_NORMAL = 0x20000, // object normals
    DIRTY_ALL_NORMALS = DIRTY_RENDER_NORMALS | DIRTY_VERTS_NORMAL | DIRTY_FACES_NORMAL | DIRTY_CORNERS_NORMAL,
    DIRTY_CACHES = DIRTY_VERTS_NORMAL | DIRTY_FACES_NORMAL | DIRTY_CORNERS_NORMAL | DIRTY_BOUNDING_BOX | DIRTY_BOUNDING_BOX_XF,
    DIRTY_ALL = 0x3FFFF
};

class MRMESH_CLASS VisualObject : public Object
{
public:
    MRMESH_API VisualObject();

    VisualObject( VisualObject&& ) = default;
    VisualObject& operator = ( VisualObject&& ) = default;
    virtual ~VisualObject() = default;

    constexpr static const char* TypeName() noexcept { return "VisualObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    // set visual property in all viewports specified by the mask
    MRMESH_API void setVisualizeProperty( bool value, unsigned type, ViewportMask viewportMask );
    // set visual property mask
    MRMESH_API virtual void setVisualizePropertyMask( unsigned type, ViewportMask viewportMask );
    // returns true if the property is set at least in one viewport specified by the mask
    MRMESH_API bool getVisualizeProperty( unsigned type, ViewportMask viewportMask ) const;
    // returns mask of viewports where given property is set
    MRMESH_API virtual const ViewportMask& getVisualizePropertyMask( unsigned type ) const;
    // toggle visual property in all viewports specified by the mask
    MRMESH_API void toggleVisualizeProperty( unsigned type, ViewportMask viewportMask );

    // get all visualize properties masks as array
    MRMESH_API virtual AllVisualizeProperties getAllVisualizeProperties() const;
    // set all visualize properties masks from array
    MRMESH_API virtual void setAllVisualizeProperties( const AllVisualizeProperties& properties );

    // shows/hides labels
    void showLabels( bool on ) { return setVisualizeProperty( on, unsigned( VisualizeMaskType::Labels ), ViewportMask::all() ); }
    bool showLabels() const { return getVisualizeProperty( unsigned( VisualizeMaskType::Labels ), ViewportMask::any() ); }

    // shows/hides name
    void showName( bool on ) { return setVisualizeProperty( on, unsigned( VisualizeMaskType::Name ), ViewportMask::all() ); }
    bool showName() const { return getVisualizeProperty( unsigned( VisualizeMaskType::Name ), ViewportMask::any() ); }

    // if selected returns color of object when it is selected
    // otherwise returns color of object when it is not selected
    MRMESH_API const Color& getFrontColor( bool selected = true ) const;
    // if selected sets color of object when it is selected
    // otherwise sets color of object when it is not selected
    MRMESH_API virtual void setFrontColor( const Color& color, bool selected = true );

    MRMESH_API const Color& getBackColor() const;
    MRMESH_API virtual void setBackColor( const Color& color );

    MRMESH_API const Color& getLabelsColor() const;
    MRMESH_API virtual void setLabelsColor( const Color& color );

    MRMESH_API virtual void setDirtyFlags( uint32_t mask );
    MRMESH_API const uint32_t& getDirtyFlags() const;

    MRMESH_API void resetDirty() const;

    MRMESH_API Box3f getBoundingBox() const;
    void setXf( const AffineXf3f& xf ) override { Object::setXf( xf ); setDirtyFlags( DIRTY_BOUNDING_BOX_XF ); };
    MRMESH_API Box3f getBoundingBoxXf() const;

    MRMESH_API const Vector<Vector3f, VertId>& getVertsNormals() const;

    virtual bool getRedrawFlag( ViewportMask viewportMask ) const override 
    {
        return Object::getRedrawFlag( viewportMask ) || 
            ( isVisible( viewportMask ) &&
              ( dirty_ & ( ~( DIRTY_CACHES ) ) ) );
    }

    // Is object pickable by gl
    bool isPickable( ViewportMask viewportMask = ViewportMask::all() ) const{return !(pickable_ & viewportMask).empty();}

    // Set object pickability by gl
    MRMESH_API virtual void setPickable( bool on, ViewportMask viewportMask = ViewportMask::all() );

    const MeshTexture& getTexture() const { return texture_; }
    virtual void setTexture( MeshTexture texture ) { texture_ = std::move( texture ); dirty_ |= DIRTY_TEXTURE; }

    const Vector<UVCoord, VertId>& getUVCoords() const { return uvCoordinates_; }
    virtual void setUVCoords( Vector<UVCoord, VertId> uvCoordinates ) { uvCoordinates_ = std::move( uvCoordinates ); dirty_ |= DIRTY_UV; }

    const Vector<Color, VertId>& getVertsColorMap() const { return vertsColorMap_; }

    virtual void setVertsColorMap( Vector<Color, VertId> vertsColorMap ) { vertsColorMap_ = std::move( vertsColorMap ); dirty_ |= DIRTY_VERTS_COLORMAP; }

    ColoringType getColoringType() const { return coloringType_; }
    MRMESH_API virtual void setColoringType( ColoringType coloringType );

    float getShininess() const { return shininess_; }
    virtual void setShininess( float shininess ) { shininess_ = shininess; needRedraw_ = true; }

    const std::vector<MeshLabel>& getLabels() const { return labels_; }
    virtual void setLabels( std::vector<MeshLabel> labels ) { labels_ = std::move( labels ); needRedraw_ = true; }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void render( const RenderParams& ) const;
    MRMESH_API virtual void renderForPicker( const BaseRenderParams&, unsigned ) const;

    // swaps this object with other
    MRMESH_API virtual void swap( Object& other ) override;

    // returns cached bounding box of this object in world coordinates;
    // if you need bounding box in local coordinates please call getBoundingBox()
    MRMESH_API virtual const Box3f getWorldBox() const;

    // this ctor is public only for std::make_shared used inside clone()
    VisualObject( ProtectedStruct, const VisualObject& obj ) : VisualObject( obj ) {}

protected:

    MRMESH_API VisualObject( const VisualObject& obj );

    // each renderable child of VisualObject should imlpement this method
    // and assign renderObj_ inside
    virtual void setupRenderObject_() const {}
    mutable std::unique_ptr<IRenderObject> renderObj_;

    // Visualization options
    // Each option is a binary mask specifying on which viewport each option is set.
    // When using a single viewport, standard boolean can still be used for simplicity.
    ViewportMask showTexture_;
    ViewportMask clipByPlane_;
    ViewportMask showLabels_;
    ViewportMask showName_;
    ViewportMask cropLabels_ = ViewportMask::all(); 
    ViewportMask pickable_ = ViewportMask::all(); // enable picking by gl    
    ViewportMask invertNormals_; // invert mesh normals
    ViewportMask depthTest_ = ViewportMask::all();

    Color labelsColor_ = Color::black();

    float shininess_{35.0f}; // specular exponent

    // Main coloring options
    ColoringType coloringType_{ColoringType::SolidColor};
    Vector<Color, VertId> vertsColorMap_;
    Color selectedColor_;
    Color unselectedColor_;
    Color backFacesColor_;

    // Texture options
    MeshTexture texture_;
    Vector<UVCoord, VertId> uvCoordinates_; // vertices coordinates in texture

    std::vector<MeshLabel> labels_;

    MRMESH_API ViewportMask& getVisualizePropertyMask_( unsigned type );

    // Marks dirty buffers that need to be uploaded to OpenGL
    mutable uint32_t dirty_{DIRTY_ALL};

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    virtual Box3f computeBoundingBox_() const { return Box3f(); }
    virtual Box3f computeBoundingBoxXf_() const { return Box3f(); }

    virtual Vector<Vector3f, VertId> computeVertsNormals_() const { return {}; }

    // adds information about bounding box in res
    MRMESH_API void boundingBoxToInfoLines_( std::vector<std::string> & res ) const;

private:
    mutable Box3f boundingBoxCache_;
    mutable Box3f boundingBoxCacheXf_;

    mutable Vector<Vector3f, VertId> vertsNormalsCache_;

    // this is private function to set default colors of this type (Visual Object) in constructor only
    void setDefaultColors_();
};

} // namespace MR
