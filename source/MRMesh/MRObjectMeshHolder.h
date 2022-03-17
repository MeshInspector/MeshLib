#pragma once

#include "MRVisualObject.h"
#include "MRXfBasedCache.h"

namespace MR
{

struct MeshVisualizePropertyType : VisualizeMaskType
{
    enum Type : unsigned
    {
        Faces = VisualizeMaskType::VisualizePropsCount,
        Edges,
        SelectedFaces,
        SelectedEdges,
        FlatShading,
        OnlyOddFragments,
        BordersHighlight,
        MeshVisualizePropsCount
    };
};

// an object that stores a mesh
class MRMESH_CLASS ObjectMeshHolder : public VisualObject
{
public:
    MRMESH_API ObjectMeshHolder();

    ObjectMeshHolder( ObjectMeshHolder&& ) noexcept = default;
    ObjectMeshHolder& operator = ( ObjectMeshHolder&& ) noexcept = default;
    virtual ~ObjectMeshHolder() = default;

    constexpr static const char* TypeName() noexcept { return "MeshHolder"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    const std::shared_ptr< const Mesh >& mesh() const
    { return reinterpret_cast< const std::shared_ptr<const Mesh>& >( mesh_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setDirtyFlags( uint32_t mask ) override;

    const FaceBitSet& getSelectedFaces() const { return selectedTriangles_; }
    MRMESH_API virtual void selectFaces( FaceBitSet newSelection );
    // returns colors of selected triangles
    MRMESH_API const Color& getSelectedFacesColor() const;
    // sets colors of selected triangles
    MRMESH_API virtual void setSelectedFacesColor( const Color& color );

    const UndirectedEdgeBitSet& getSelectedEdges() const { return selectedEdges_; }
    MRMESH_API virtual void selectEdges( const UndirectedEdgeBitSet& newSelection );
    // returns colors of selected edges
    MRMESH_API const Color& getSelectedEdgesColor() const;
    // sets colors of selected edges
    MRMESH_API virtual void setSelectedEdgesColor( const Color& color );

    // Edges on mesh, that will have sharp visualization even with smooth shading
    const UndirectedEdgeBitSet& creases() const { return creases_; }
    MRMESH_API virtual void setCreases( UndirectedEdgeBitSet creases );

    // sets flat (true) or smooth (false) shading
    void setFlatShading( bool on )
    {
        return setVisualizeProperty( on, unsigned( MeshVisualizePropertyType::FlatShading ), ViewportMask::all() );
    }
    bool flatShading() const
    {
        return getVisualizeProperty( unsigned( MeshVisualizePropertyType::FlatShading ), ViewportMask::any() );
    }

    // get all visualize properties masks as array
    MRMESH_API virtual AllVisualizeProperties getAllVisualizeProperties() const override;
    // returns mask of viewports where given property is set
    MRMESH_API virtual const ViewportMask& getVisualizePropertyMask( unsigned type ) const override;

    MRMESH_API const Vector<Vector3f, FaceId>& getFacesNormals() const;
    MRMESH_API const Vector<TriangleCornerNormals, FaceId>& getCornerNormals() const;

    const Vector<Color, FaceId>& getFacesColorMap() const
    {
        return facesColorMap_;
    }
    virtual void setFacesColorMap( Vector<Color, FaceId> facesColorMap )
    {
        facesColorMap_ = std::move( facesColorMap ); dirty_ |= DIRTY_PRIMITIVE_COLORMAP;
    }

    float getEdgeWidth() const
    {
        return edgeWidth_;
    }
    virtual void setEdgeWidth( float edgeWidth )
    {
        edgeWidth_ = edgeWidth; needRedraw_ = true;
    }

    const Color& getEdgesColor() const
    {
        return edgesColor_;
    }
    virtual void setEdgesColor( const Color& color )
    {
        edgesColor_ = color; needRedraw_ = true;
    }

    const Color& getBordersColor() const
    {
        return bordersColor_;
    }
    virtual void setBordersColor( const Color& color )
    {
        bordersColor_ = color; needRedraw_ = true;
    }

    // this ctor is public only for std::make_shared used inside clone()
    ObjectMeshHolder( ProtectedStruct, const ObjectMeshHolder& obj ) : ObjectMeshHolder( obj )
    {}

    // returns dirty flag of currently using normal type if they are dirty in render representation
    MRMESH_API uint32_t getNeededNormalsRenderDirtyValue( ViewportMask viewportMask ) const;

    MRMESH_API virtual bool getRedrawFlag( ViewportMask viewportMask ) const override;

    // reset dirty flags without some specific bits (useful for lazy normals update)
    MRMESH_API virtual void resetDirtyExeptMask( uint32_t mask ) const;
public:
    // returns cached information whether the mesh is closed
    MRMESH_API bool isMeshClosed() const;
    // returns cached bounding box of this mesh object in world coordinates;
    // if you need bounding box in local coordinates please call getBoundingBox()
    MRMESH_API virtual Box3f getWorldBox() const override;
    // returns cached information about the number of selected faces in the mesh
    MRMESH_API size_t numSelectedFaces() const;
    // returns cached information about the number of selected undirected edges in the mesh
    MRMESH_API size_t numSelectedEdges() const;
    // returns cached information about the number of crease undirected edges in the mesh
    MRMESH_API size_t numCreaseEdges() const;
    // returns cached summed area of mesh triangles
    MRMESH_API double totalArea() const;
protected:
    FaceBitSet selectedTriangles_;
    UndirectedEdgeBitSet selectedEdges_;
    UndirectedEdgeBitSet creases_;

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
    mutable std::optional<size_t> numSelectedFaces_, numSelectedEdges_, numCreaseEdges_;
    mutable std::optional<double> totalArea_;
    mutable XfBasedCache<Box3f> worldBox_;

    MRMESH_API ObjectMeshHolder( const ObjectMeshHolder& other );

    // swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

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
    ViewportMask showSelectedFaces_ = ViewportMask::all();
    ViewportMask showBordersHighlight_;
    ViewportMask flatShading_; // toggle per-face or per-vertex properties
    ViewportMask onlyOddFragments_;

    Color edgesColor_ = Color::black();
    Color bordersColor_ = Color::black();
    Color edgeSelectionColor_ = Color::black();
    Color faceSelectionColor_;

    Vector<Color, FaceId> facesColorMap_;
    float edgeWidth_{ 0.5f };

    std::shared_ptr<Mesh> mesh_;
private:
    // this is private function to set default colors of this type (ObjectMeshHolder) in constructor only
    void setDefaultColors_();
};

} //namespace MR

