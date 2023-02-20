#pragma once

#include "MRVisualObject.h"
#include "MRXfBasedCache.h"
#include "MRMeshPart.h"

namespace MR
{

struct MeshVisualizePropertyType : VisualizeMaskType
{
    enum Type : unsigned
    {
        Faces = VisualizeMaskType::VisualizePropsCount,
        Texture,
        Edges,
        SelectedFaces,
        SelectedEdges,
        FlatShading,
        OnlyOddFragments,
        BordersHighlight,
        MeshVisualizePropsCount
    };
};

/// an object that stores a mesh
/// \ingroup ModelHolderGroup
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

    MRMESH_API virtual bool hasVisualRepresentation() const override;

    const std::shared_ptr< const Mesh >& mesh() const
    { return reinterpret_cast< const std::shared_ptr<const Mesh>& >( mesh_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    /// \return the pair ( mesh, selected triangles ) if any triangle is selected or whole mesh otherwise
    MeshPart meshPart() const { return selectedTriangles_.any() ? MeshPart{ *mesh_, &selectedTriangles_ } : *mesh_; }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setDirtyFlags( uint32_t mask ) override;

    const FaceBitSet& getSelectedFaces() const { return selectedTriangles_; }
    MRMESH_API virtual void selectFaces( FaceBitSet newSelection );
    /// returns colors of selected triangles
    MRMESH_API const Color& getSelectedFacesColor( ViewportId id = {} ) const;
    /// sets colors of selected triangles
    MRMESH_API virtual void setSelectedFacesColor( const Color& color, ViewportId id = {} );

    const UndirectedEdgeBitSet& getSelectedEdges() const { return selectedEdges_; }
    MRMESH_API virtual void selectEdges( UndirectedEdgeBitSet newSelection );
    /// returns colors of selected edges
    MRMESH_API const Color& getSelectedEdgesColor( ViewportId id = {} ) const;
    /// sets colors of selected edges
    MRMESH_API virtual void setSelectedEdgesColor( const Color& color, ViewportId id = {} );

    MRMESH_API const ViewportProperty<Color>& getSelectedEdgesColorsForAllViewports() const;
    MRMESH_API virtual void setSelectedEdgesColorsForAllViewports( ViewportProperty<Color> val );

    MRMESH_API const ViewportProperty<Color>& getSelectedFacesColorsForAllViewports() const;
    MRMESH_API virtual void setSelectedFacesColorsForAllViewports( ViewportProperty<Color> val );

    MRMESH_API const ViewportProperty<Color>& getEdgesColorsForAllViewports() const;
    MRMESH_API virtual void setEdgesColorsForAllViewports( ViewportProperty<Color> val );

    MRMESH_API const ViewportProperty<Color>& getBordersColorsForAllViewports() const;
    MRMESH_API virtual void setBordersColorsForAllViewports( ViewportProperty<Color> val );

    /// Edges on mesh, that will have sharp visualization even with smooth shading
    const UndirectedEdgeBitSet& creases() const { return creases_; }
    MRMESH_API virtual void setCreases( UndirectedEdgeBitSet creases );

    /// sets flat (true) or smooth (false) shading
    void setFlatShading( bool on )
    { return setVisualizeProperty( on, unsigned( MeshVisualizePropertyType::FlatShading ), ViewportMask::all() ); }
    bool flatShading() const
    { return getVisualizeProperty( unsigned( MeshVisualizePropertyType::FlatShading ), ViewportMask::any() ); }

    /// get all visualize properties masks as array
    MRMESH_API virtual AllVisualizeProperties getAllVisualizeProperties() const override;
    /// returns mask of viewports where given property is set
    MRMESH_API virtual const ViewportMask& getVisualizePropertyMask( unsigned type ) const override;

    const Vector<Color, FaceId>& getFacesColorMap() const { return facesColorMap_; }
    virtual void setFacesColorMap( Vector<Color, FaceId> facesColorMap )
    { facesColorMap_ = std::move( facesColorMap ); dirty_ |= DIRTY_PRIMITIVE_COLORMAP; }

    float getEdgeWidth() const { return edgeWidth_; }
    virtual void setEdgeWidth( float edgeWidth )
    { edgeWidth_ = edgeWidth; needRedraw_ = true; }

    const Color& getEdgesColor( ViewportId id = {} ) const { return edgesColor_.get(id); }
    virtual void setEdgesColor( const Color& color, ViewportId id = {} )
    { edgesColor_.set( color, id ); needRedraw_ = true; }

    const Color& getBordersColor( ViewportId id = {} ) const { return bordersColor_.get( id ); }
    virtual void setBordersColor( const Color& color, ViewportId id = {} )
    { bordersColor_.set( color, id ); needRedraw_ = true; }

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectMeshHolder( ProtectedStruct, const ObjectMeshHolder& obj ) : ObjectMeshHolder( obj )
    {}

    const MeshTexture& getTexture() const { return texture_; }
    virtual void setTexture( MeshTexture texture ) { texture_ = std::move( texture ); dirty_ |= DIRTY_TEXTURE; }

    const Vector<UVCoord, VertId>& getUVCoords() const { return uvCoordinates_; }
    virtual void setUVCoords( Vector<UVCoord, VertId> uvCoordinates ) { uvCoordinates_ = std::move( uvCoordinates ); dirty_ |= DIRTY_UV; }
    void updateUVCoords( Vector<UVCoord, VertId>& updated ) { std::swap( uvCoordinates_, updated ); dirty_ |= DIRTY_UV; }

    // ancillary texture can be used to have custom features visualization without affecting real one
    const MeshTexture& getAncillaryTexture() const { return ancillaryTexture_; }
    virtual void setAncillaryTexture( MeshTexture texture ) { ancillaryTexture_ = std::move( texture ); dirty_ |= DIRTY_TEXTURE; }

    const Vector<UVCoord, VertId>& getAncillaryUVCoords() const { return ancillaryUVCoordinates_; }
    virtual void setAncillaryUVCoords( Vector<UVCoord, VertId> uvCoordinates ) { ancillaryUVCoordinates_ = std::move( uvCoordinates ); dirty_ |= DIRTY_UV; }
    void updateAncillaryUVCoords( Vector<UVCoord, VertId>& updated ) { std::swap( ancillaryUVCoordinates_, updated ); dirty_ |= DIRTY_UV; }
    
    bool hasAncillaryTexture() const { return !ancillaryUVCoordinates_.empty(); }

    /// returns dirty flag of currently using normal type if they are dirty in render representation
    MRMESH_API uint32_t getNeededNormalsRenderDirtyValue( ViewportMask viewportMask ) const;

    MRMESH_API virtual bool getRedrawFlag( ViewportMask viewportMask ) const override;

    /// reset dirty flags without some specific bits (useful for lazy normals update)
    MRMESH_API virtual void resetDirtyExeptMask( uint32_t mask ) const;

    /// returns cached information whether the mesh is closed
    MRMESH_API bool isMeshClosed() const;
    /// returns cached bounding box of this mesh object in world coordinates;
    /// if you need bounding box in local coordinates please call getBoundingBox()
    MRMESH_API virtual Box3f getWorldBox( ViewportId = {} ) const override;
    /// returns cached information about the number of selected faces in the mesh
    MRMESH_API size_t numSelectedFaces() const;
    /// returns cached information about the number of selected undirected edges in the mesh
    MRMESH_API size_t numSelectedEdges() const;
    /// returns cached information about the number of crease undirected edges in the mesh
    MRMESH_API size_t numCreaseEdges() const;
    /// returns cached summed area of mesh triangles
    MRMESH_API double totalArea() const;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

    /// returns cached information about the number of holes in the mesh
    MRMESH_API size_t numHoles() const;

protected:
    FaceBitSet selectedTriangles_;
    UndirectedEdgeBitSet selectedEdges_;
    UndirectedEdgeBitSet creases_;

    /// Texture options
    MeshTexture texture_;
    Vector<UVCoord, VertId> uvCoordinates_; ///< vertices coordinates in texture

    MeshTexture ancillaryTexture_;
    Vector<UVCoord, VertId> ancillaryUVCoordinates_; ///< vertices coordinates in ancillary texture

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
    mutable ViewportProperty<XfBasedCache<Box3f>> worldBox_;

    MRMESH_API ObjectMeshHolder( const ObjectMeshHolder& other );

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual tl::expected<std::future<void>, std::string> serializeModel_( const std::filesystem::path& path ) const override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API tl::expected<void, std::string> deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} ) override;

    MRMESH_API virtual Box3f computeBoundingBox_() const override;

    MRMESH_API virtual void setupRenderObject_() const override;

    MRMESH_API virtual void updateMeshStat_() const;

    ViewportMask showTexture_;
    ViewportMask showFaces_ = ViewportMask::all();
    ViewportMask showEdges_;
    ViewportMask showSelectedEdges_ = ViewportMask::all();
    ViewportMask showSelectedFaces_ = ViewportMask::all();
    ViewportMask showBordersHighlight_;
    ViewportMask flatShading_; ///< toggle per-face or per-vertex properties
    ViewportMask onlyOddFragments_;

    ViewportProperty<Color> edgesColor_;
    ViewportProperty<Color> bordersColor_;
    ViewportProperty<Color> edgeSelectionColor_;
    ViewportProperty<Color> faceSelectionColor_;

    Vector<Color, FaceId> facesColorMap_;
    float edgeWidth_{ 0.5f };

    std::shared_ptr<Mesh> mesh_;

private:
    /// this is private function to set default colors of this type (ObjectMeshHolder) in constructor only
    void setDefaultColors_();
};

} // namespace MR

