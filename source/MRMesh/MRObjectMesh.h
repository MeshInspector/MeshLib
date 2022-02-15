#pragma once

#include "MRVisualObject.h"

namespace MR
{

struct MeshVisualizePropertyType : VisualizeMaskType
{
    enum Type : unsigned
    {
        Faces = VisualizeMaskType::VisualizePropsCount,
        Edges,
        SelectedEdges,
        FlatShading,
        OnlyOddFragments,
        BordersHighlight,
        MeshVisualizePropsCount
    };
};

// an object that stores a mesh
class MRMESH_CLASS ObjectMesh : public VisualObject
{
public:
    MRMESH_API ObjectMesh();

    ObjectMesh( ObjectMesh&& ) noexcept = default;
    ObjectMesh& operator = ( ObjectMesh&& ) noexcept = default;
    virtual ~ObjectMesh() = default;

    constexpr static const char* TypeName() noexcept { return "ObjectMesh"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    // returns variable mesh, if const mesh is needed use `mesh()` instead
    virtual const std::shared_ptr< Mesh > & varMesh() { return mesh_; }
    // returns const mesh, if variable mesh is needed use `varMesh()` instead
    const std::shared_ptr< const Mesh >& mesh() const
    { return reinterpret_cast<const std::shared_ptr<const Mesh>&>( mesh_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    // sets given mesh to this
    MRMESH_API virtual void setMesh( std::shared_ptr< Mesh > mesh );
    // sets given mesh to this, and returns back previous mesh of this
    MRMESH_API virtual void swapMesh( std::shared_ptr< Mesh > & mesh );
    void setXf( const AffineXf3f& xf ) override { VisualObject::setXf( xf ); worldBox_.reset(); }

    const FaceBitSet& getSelectedFaces() const
    {
        return selectedTriangles_;
    }
    MRMESH_API virtual void selectFaces( FaceBitSet newSelection );
    // returns colors of selected triangles
    MRMESH_API const Color& getSelectedFacesColor() const;
    // sets colors of selected triangles
    MRMESH_API virtual void setSelectedFacesColor( const Color& color );
    // returns true if selected faces are shown now, false otherwise
    bool areSelectedFacesShown() const { return showSelectedFaces_; }
    // turn selected faces coloring on/off
    MRMESH_API virtual void showSelectedFaces( bool show );

    const UndirectedEdgeBitSet& getSelectedEdges() const
    {
        return selectedEdges_;
    }
    MRMESH_API virtual void selectEdges( const UndirectedEdgeBitSet& newSelection );
    // returns colors of selected edges
    MRMESH_API const Color& getSelectedEdgesColor() const;
    // sets colors of selected edges
    MRMESH_API virtual void setSelectedEdgesColor( const Color& color );

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setDirtyFlags( uint32_t mask ) override;

    // Edges on mesh, that will have sharp visualization even with smooth shading
    const UndirectedEdgeBitSet& creases() const { return creases_; }
    MRMESH_API virtual void setCreases( UndirectedEdgeBitSet creases );

    // sets flat (true) or smooth (false) shading
    void setFlatShading( bool on ) { return setVisualizeProperty( on, unsigned( MeshVisualizePropertyType::FlatShading ), ViewportMask::all() ); }
    bool flatShading() const { return getVisualizeProperty( unsigned( MeshVisualizePropertyType::FlatShading ), ViewportMask::any() ); }

    // get all visualize properties masks as array
    MRMESH_API virtual AllVisualizeProperties getAllVisualizeProperties() const override;
    // returns mask of viewports where given property is set
    MRMESH_API virtual const ViewportMask& getVisualizePropertyMask( unsigned type ) const override;

    MRMESH_API const Vector<Vector3f, FaceId>& getFacesNormals() const;
    MRMESH_API const Vector<TriangleCornerNormals, FaceId>& getCornerNormals() const;

    const Vector<Color, FaceId>& getFacesColorMap() const { return facesColorMap_; }
    virtual void setFacesColorMap( Vector<Color, FaceId> facesColorMap ) { facesColorMap_ = std::move( facesColorMap ); dirty_ |= DIRTY_PRIMITIVE_COLORMAP; }

    float getEdgeWidth() const { return edgeWidth_; }
    virtual void setEdgeWidth( float edgeWidth ) { edgeWidth_ = edgeWidth; needRedraw_ = true; }

    const Color& getEdgesColor() const { return edgesColor_; }
    virtual void setEdgesColor( const Color& color ) { edgesColor_ = color; needRedraw_ = true; }

    const Color& getBordersColor() const { return bordersColor_; }
    virtual void setBordersColor( const Color& color ) { bordersColor_ = color; needRedraw_ = true; }

    // swaps this object with other
    // note: do not swap object signals, so listeners will get notifications from swapped object
    MRMESH_API virtual void swap( Object& other ) override;

    // this ctor is public only for std::make_shared used inside clone()
    ObjectMesh( ProtectedStruct, const ObjectMesh& obj ) : ObjectMesh( obj ) {}

    // returns dirty flag of currently using normal type if they are dirty in render representation
    MRMESH_API uint32_t getNeededNormalsRenderDirtyValue( ViewportMask viewportMask ) const;

    MRMESH_API virtual bool getRedrawFlag( ViewportMask viewportMask ) const override;

    // reset dirty flags without some specific bits (useful for lazy normals update)
    MRMESH_API virtual void resetDirtyExeptMask( uint32_t mask ) const;

    // given ray in world coordinates, e.g. obtained from Viewport::unprojectPixelRay;
    // finds its intersection with the mesh of this object considering its transformation relative to the world;
    // it is inefficient to call this function for many rays, because it computes world-to-local xf every time
    MRMESH_API std::optional<MeshIntersectionResult> worldRayIntersection( const Line3f& worldRay, const FaceBitSet* region = nullptr ) const;

    // returns cached information whether the mesh is closed
    MRMESH_API bool isMeshClosed() const;
    // returns cached bounding box of this mesh object in world coordinates;
    // if you need bounding box in local coordinates please call getBoundingBox()
    MRMESH_API const Box3f getWorldBox() const;
    // returns cached information about the number of selected faces in the mesh
    MRMESH_API size_t numSelectedFaces() const;
    // returns cached information about the number of selected undirected edges in the mesh
    MRMESH_API size_t numSelectedEdges() const;

    // signal about mesh changing, triggered in setDirtyFlag
    using MeshChangedSignal = boost::signals2::signal<void( uint32_t mask )>;
    MeshChangedSignal meshChangedSignal;
private:
    FaceBitSet selectedTriangles_;
    UndirectedEdgeBitSet selectedEdges_;
    UndirectedEdgeBitSet creases_;

    Color faceSelectionColor_;

    bool showSelectedFaces_{true};
    // this is private function to set default colors of this type (ObjectMesh) in constructor only
    void setDefaultColors_();

    mutable Vector<TriangleCornerNormals, FaceId> cornerNormalsCache_;
    mutable Vector<Vector3f, FaceId> facesNormalsCache_;

    struct MeshStat
    {
        size_t numComponents = 0;
        size_t numUndirectedEdges = 0;
        size_t numHoles = 0;
    };
    mutable std::optional<MeshStat> meshStat_;
    mutable std::optional<bool> meshIsClosed_;
    mutable std::optional<size_t> numSelectedFaces_, numSelectedEdges_;
    mutable std::optional<Box3f> worldBox_;

protected:
    MRMESH_API ObjectMesh( const ObjectMesh& other );

    MRMESH_API virtual tl::expected<std::future<void>, std::string> serializeModel_( const std::filesystem::path& path ) const override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API tl::expected<void, std::string> deserializeModel_( const std::filesystem::path& path ) override;

    MRMESH_API virtual Box3f computeBoundingBox_() const override;
    MRMESH_API virtual Box3f computeBoundingBoxXf_() const override;

    MRMESH_API virtual Vector<Vector3f, VertId> computeVertsNormals_() const override;
    MRMESH_API virtual Vector<Vector3f, FaceId> computeFacesNormals_() const;
    MRMESH_API virtual Vector<TriangleCornerNormals, FaceId> computeCornerNormals_() const;

    MRMESH_API virtual void setupRenderObject_() const override;

    ViewportMask showFaces_ = ViewportMask::all();
    ViewportMask showEdges_;
    ViewportMask showSelectedEdges_ = ViewportMask::all();
    ViewportMask showBordersHighlight_;
    ViewportMask flatShading_; // toggle per-face or per-vertex properties
    ViewportMask onlyOddFragments_;

    Color edgesColor_ = Color::black();
    Color bordersColor_ = Color::black();
    Color edgeSelectionColor_ = Color::black();

    Vector<Color, FaceId> facesColorMap_;
    float edgeWidth_{ 0.5f };

    std::shared_ptr<Mesh> mesh_;
};

} //namespace MR

