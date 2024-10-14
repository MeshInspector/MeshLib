#pragma once

#include "MRVisualObject.h"
#include "MRXfBasedCache.h"
#include "MRMeshPart.h"

namespace MR
{

enum class MRMESH_CLASS MeshVisualizePropertyType
{
    Faces,
    Texture,
    Edges,
    Points,
    SelectedFaces,
    SelectedEdges,
    EnableShading,
    FlatShading,
    OnlyOddFragments,
    BordersHighlight,
    PolygonOffsetFromCamera, // recommended for drawing edges on top of mesh
    _count [[maybe_unused]],
};
template <> struct IsVisualizeMaskEnum<MeshVisualizePropertyType> : std::true_type {};

/// an object that stores a mesh
/// \ingroup ModelHolderGroup
class MRMESH_CLASS ObjectMeshHolder : public VisualObject
{
public:
    MRMESH_API ObjectMeshHolder();

    ObjectMeshHolder( ObjectMeshHolder&& ) noexcept = default;
    ObjectMeshHolder& operator = ( ObjectMeshHolder&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "MeshHolder"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    /// mesh object can be seen if the mesh has at least one edge
    MRMESH_API virtual bool hasVisualRepresentation() const override;

    [[nodiscard]] virtual bool hasModel() const override { return bool( mesh_ ); }

    const std::shared_ptr< const Mesh >& mesh() const
    { return reinterpret_cast< const std::shared_ptr<const Mesh>& >( mesh_ ); } // reinterpret_cast to avoid making a copy of shared_ptr

    /// \return the pair ( mesh, selected triangles ) if any triangle is selected or whole mesh otherwise
    MeshPart meshPart() const { return selectedTriangles_.any() ? MeshPart{ *mesh_, &selectedTriangles_ } : *mesh_; }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setDirtyFlags( uint32_t mask, bool invalidateCaches = true ) override;

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
    { return setVisualizeProperty( on, MeshVisualizePropertyType::FlatShading, ViewportMask::all() ); }
    bool flatShading() const
    { return getVisualizeProperty( MeshVisualizePropertyType::FlatShading, ViewportMask::any() ); }

    [[nodiscard]] MRMESH_API bool supportsVisualizeProperty( AnyVisualizeMaskEnum type ) const override;

    /// get all visualize properties masks
    MRMESH_API AllVisualizeProperties getAllVisualizeProperties() const override;
    /// returns mask of viewports where given property is set
    MRMESH_API const ViewportMask& getVisualizePropertyMask( AnyVisualizeMaskEnum type ) const override;

    const FaceColors& getFacesColorMap() const { return facesColorMap_; }
    virtual void setFacesColorMap( FaceColors facesColorMap )
    { facesColorMap_ = std::move( facesColorMap ); dirty_ |= DIRTY_PRIMITIVE_COLORMAP; }
    virtual void updateFacesColorMap( FaceColors& updated )
    { std::swap( facesColorMap_, updated ); dirty_ |= DIRTY_PRIMITIVE_COLORMAP; }

    MRMESH_API virtual void setEdgeWidth( float edgeWidth );
    float getEdgeWidth() const { return edgeWidth_; }
    MRMESH_API virtual void setPointSize( float size );
    virtual float getPointSize() const { return pointSize_; }

    const Color& getEdgesColor( ViewportId id = {} ) const { return edgesColor_.get(id); }
    virtual void setEdgesColor( const Color& color, ViewportId id = {} )
    { edgesColor_.set( color, id ); needRedraw_ = true; }
    
    const Color& getPointsColor( ViewportId id = {} ) const { return pointsColor_.get(id); }
    virtual void setPointsColor( const Color& color, ViewportId id = {} )
    { pointsColor_.set( color, id ); needRedraw_ = true; }

    const Color& getBordersColor( ViewportId id = {} ) const { return bordersColor_.get( id ); }
    virtual void setBordersColor( const Color& color, ViewportId id = {} )
    { bordersColor_.set( color, id ); needRedraw_ = true; }

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectMeshHolder( ProtectedStruct, const ObjectMeshHolder& obj ) : ObjectMeshHolder( obj )
    {}

    /// returns first texture in the vector. If there is no textures, returns empty texture
    MRMESH_API const MeshTexture& getTexture() const;
    // for backward compatibility
    [[deprecated]] MRMESH_API virtual void setTexture( MeshTexture texture );
    [[deprecated]] MRMESH_API virtual void updateTexture( MeshTexture& updated );
    const Vector<MeshTexture, TextureId>& getTextures() const { return textures_; }
    virtual void setTextures( Vector<MeshTexture, TextureId> texture ) { textures_ = std::move( texture ); dirty_ |= DIRTY_TEXTURE; }
    virtual void updateTextures( Vector<MeshTexture, TextureId>& updated ) { std::swap( textures_, updated ); dirty_ |= DIRTY_TEXTURE; }

    /// the texture ids for the faces if more than one texture is used to texture the object
    /// texture coordinates (uvCoordinates_) at a point can belong to different textures, depending on which face the point belongs to
    virtual void setTexturePerFace( Vector<TextureId, FaceId> texturePerFace ) { texturePerFace_ = std::move( texturePerFace ); dirty_ |= DIRTY_TEXTURE_PER_FACE; }
    virtual void updateTexturePerFace( Vector<TextureId, FaceId>& texturePerFace ) { std::swap( texturePerFace_, texturePerFace ); dirty_ |= DIRTY_TEXTURE_PER_FACE; }
    virtual void addTexture( MeshTexture texture ) { textures_.emplace_back( std::move( texture ) ); dirty_ |= DIRTY_TEXTURE_PER_FACE; }
    const TexturePerFace& getTexturePerFace() const { return texturePerFace_; }
    
    const VertUVCoords& getUVCoords() const { return uvCoordinates_; }
    virtual void setUVCoords( VertUVCoords uvCoordinates ) { uvCoordinates_ = std::move( uvCoordinates ); dirty_ |= DIRTY_UV; }
    virtual void updateUVCoords( VertUVCoords& updated ) { std::swap( uvCoordinates_, updated ); dirty_ |= DIRTY_UV; }

    /// copies texture, UV-coordinates and vertex colors from given source object \param src using given map \param thisToSrc
    MRMESH_API virtual void copyTextureAndColors( const ObjectMeshHolder& src, const VertMap& thisToSrc, const FaceMap& thisToSrcFaces = {} );

    MRMESH_API void copyColors( const VisualObject& src, const VertMap& thisToSrc, const FaceMap& thisToSrcFaces = {} ) override;

    // ancillary texture can be used to have custom features visualization without affecting real one
    const MeshTexture& getAncillaryTexture() const { return ancillaryTexture_; }
    virtual void setAncillaryTexture( MeshTexture texture ) { ancillaryTexture_ = std::move( texture ); dirty_ |= DIRTY_TEXTURE; }

    const VertUVCoords& getAncillaryUVCoords() const { return ancillaryUVCoordinates_; }
    virtual void setAncillaryUVCoords( VertUVCoords uvCoordinates ) { ancillaryUVCoordinates_ = std::move( uvCoordinates ); dirty_ |= DIRTY_UV; }
    void updateAncillaryUVCoords( VertUVCoords& updated ) { std::swap( ancillaryUVCoordinates_, updated ); dirty_ |= DIRTY_UV; }

    bool hasAncillaryTexture() const { return !ancillaryUVCoordinates_.empty() && !ancillaryTexture_.pixels.empty(); }
    MRMESH_API void clearAncillaryTexture();

    /// returns dirty flag of currently using normal type if they are dirty in render representation
    MRMESH_API uint32_t getNeededNormalsRenderDirtyValue( ViewportMask viewportMask ) const;

    MRMESH_API virtual bool getRedrawFlag( ViewportMask viewportMask ) const override;

    /// reset dirty flags without some specific bits (useful for lazy normals update)
    MRMESH_API virtual void resetDirtyExeptMask( uint32_t mask ) const;

    /// returns cached information whether the mesh is closed
    [[nodiscard]] MRMESH_API bool isMeshClosed() const;

    /// returns cached bounding box of this mesh object in world coordinates;
    /// if you need bounding box in local coordinates please call getBoundingBox()
    [[nodiscard]] MRMESH_API virtual Box3f getWorldBox( ViewportId = {} ) const override;

    /// returns cached information about the number of selected faces in the mesh
    [[nodiscard]] MRMESH_API size_t numSelectedFaces() const;

    /// returns cached information about the number of selected undirected edges in the mesh
    [[nodiscard]] MRMESH_API size_t numSelectedEdges() const;

    /// returns cached information about the number of crease undirected edges in the mesh
    [[nodiscard]] MRMESH_API size_t numCreaseEdges() const;

    /// returns cached summed area of mesh triangles
    [[nodiscard]] MRMESH_API double totalArea() const;

    /// returns cached area of selected triangles
    [[nodiscard]] MRMESH_API double selectedArea() const;

    /// returns cached volume of space surrounded by the mesh, which is valid only if mesh is closed
    [[nodiscard]] MRMESH_API double volume() const;

    /// returns cached average edge length
    [[nodiscard]] MRMESH_API float avgEdgeLen() const;

    /// returns cached information about the number of undirected edges in the mesh
    [[nodiscard]] MRMESH_API size_t numUndirectedEdges() const;

    /// returns cached information about the number of holes in the mesh
    [[nodiscard]] MRMESH_API size_t numHoles() const;

    /// returns cached information about the number of components in the mesh
    [[nodiscard]] MRMESH_API size_t numComponents() const;

    /// returns cached information about the number of handles in the mesh
    [[nodiscard]] MRMESH_API size_t numHandles() const;

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

    /// returns file extension used to serialize the mesh
    [[nodiscard]] const char * saveMeshFormat() const { return saveMeshFormat_; }

    /// sets file extension used to serialize the mesh: must be not null and must start from '.'
    MRMESH_API void setSaveMeshFormat( const char * newFormat );

    /// signal about face selection changing, triggered in selectFaces
    using SelectionChangedSignal = Signal<void()>;
    SelectionChangedSignal faceSelectionChangedSignal;
    SelectionChangedSignal edgeSelectionChangedSignal;
    SelectionChangedSignal creasesChangedSignal;

protected:
    FaceBitSet selectedTriangles_;
    UndirectedEdgeBitSet selectedEdges_;
    UndirectedEdgeBitSet creases_;

    /// Texture options
    Vector<MeshTexture, TextureId> textures_;
    VertUVCoords uvCoordinates_; ///< vertices coordinates in texture

    Vector<TextureId, FaceId> texturePerFace_;

    MeshTexture ancillaryTexture_;
    VertUVCoords ancillaryUVCoordinates_; ///< vertices coordinates in ancillary texture

    mutable std::optional<size_t> numHoles_;
    mutable std::optional<size_t> numComponents_;
    mutable std::optional<size_t> numUndirectedEdges_;
    mutable std::optional<size_t> numHandles_;
    mutable std::optional<bool> meshIsClosed_;
    mutable std::optional<size_t> numSelectedFaces_, numSelectedEdges_, numCreaseEdges_;
    mutable std::optional<double> totalArea_, selectedArea_;
    mutable std::optional<double> volume_;
    mutable std::optional<float> avgEdgeLen_;
    mutable ViewportProperty<XfBasedCache<Box3f>> worldBox_;

    ObjectMeshHolder( const ObjectMeshHolder& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;
    /// swaps signals, used in `swap` function to return back signals after `swapBase_`
    /// pls call Parent::swapSignals_ first when overriding this function
    MRMESH_API virtual void swapSignals_( Object& other ) override;

    MRMESH_API virtual Expected<std::future<Expected<void>>> serializeModel_( const std::filesystem::path& path ) const override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API Expected<void> deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} ) override;

    /// set all visualize properties masks
    MRMESH_API void setAllVisualizeProperties_( const AllVisualizeProperties& properties, std::size_t& pos ) override;

    MRMESH_API virtual Box3f computeBoundingBox_() const override;

    MRMESH_API virtual void setupRenderObject_() const override;

    ViewportMask showTexture_;
    ViewportMask showFaces_ = ViewportMask::all();
    ViewportMask showEdges_;
    ViewportMask showPoints_;
    ViewportMask showSelectedEdges_ = ViewportMask::all();
    ViewportMask showSelectedFaces_ = ViewportMask::all();
    ViewportMask showBordersHighlight_;
    ViewportMask polygonOffset_;
    ViewportMask flatShading_; ///< toggle per-face or per-vertex properties

    // really it shoud be one enum Shading {None, Flat, Smooth, Crease}
    // but for back capability it is easier to add global flag
    ViewportMask shadingEnabled_ = ViewportMask::all();

    ViewportMask onlyOddFragments_;

    ViewportProperty<Color> edgesColor_;
    ViewportProperty<Color> pointsColor_;
    ViewportProperty<Color> bordersColor_;
    ViewportProperty<Color> edgeSelectionColor_;
    ViewportProperty<Color> faceSelectionColor_;

    FaceColors facesColorMap_;
    float edgeWidth_{ 0.5f };
    float pointSize_{ 5.f };

    std::shared_ptr<Mesh> mesh_;

private:
    /// this is private function to set default colors of this type (ObjectMeshHolder) in constructor only
    void setDefaultColors_();

    /// set default scene-related properties
    void setDefaultSceneProperties_();

    // falls back to the internal format if no CTM format support is available
    // NOTE: CTM format support is available in the MRIOExtras library; make sure to load it if you prefer CTM
    const char * saveMeshFormat_ = ".ctm";
};

} // namespace MR
